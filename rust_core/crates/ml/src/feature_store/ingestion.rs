//! # FEATURE INGESTION - High-throughput feature pipeline
//! Drew (Data Lead): "Ingesting millions of features per second"

use super::*;
use tokio::sync::mpsc;
use std::collections::VecDeque;

/// Feature ingestion pipeline with batching and optimization
pub struct FeatureIngestionPipeline {
    store: Arc<FeatureStore>,
    
    /// Ingestion queue
    queue: Arc<RwLock<VecDeque<IngestionRequest>>>,
    
    /// Batch processor
    batch_processor: Arc<BatchProcessor>,
    
    /// Feature validators
    validators: Arc<Vec<Box<dyn FeatureValidator>>>,
    
    /// Transformation pipeline
    transformers: Arc<Vec<Box<dyn FeatureTransformer>>>,
}

/// Ingestion request
#[derive(Debug, Clone)]
pub struct IngestionRequest {
    pub entity_id: String,
    pub entity_type: String,
    pub features: HashMap<String, RawFeatureValue>,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Raw feature value before transformation
#[derive(Debug, Clone)]
pub enum RawFeatureValue {
    Float(f64),
    Integer(i64),
    String(String),
    Boolean(bool),
    Json(serde_json::Value),
    Binary(Vec<u8>),
}

/// Feature validator trait
#[async_trait]
pub trait FeatureValidator: Send + Sync {
    async fn validate(&self, feature_name: &str, value: &RawFeatureValue) -> Result<(), ValidationError>;
}

/// Feature transformer trait
#[async_trait]
pub trait FeatureTransformer: Send + Sync {
    async fn transform(&self, feature_name: &str, value: RawFeatureValue) -> Result<FeatureValue, TransformError>;
}

/// Batch processor for efficient database writes
pub struct BatchProcessor {
    batch_size: usize,
    flush_interval: Duration,
    max_retries: u32,
}

/// Validation error
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid value type for {0}")]
    InvalidType(String),
    
    #[error("Value out of range: {0}")]
    OutOfRange(String),
    
    #[error("Missing required feature: {0}")]
    MissingRequired(String),
    
    #[error("Schema violation: {0}")]
    SchemaViolation(String),
}

/// Transform error
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("Transform failed: {0}")]
    Failed(String),
    
    #[error("Unsupported transformation")]
    Unsupported,
}

impl FeatureIngestionPipeline {
    pub fn new(store: Arc<FeatureStore>) -> Self {
        Self {
            store,
            queue: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            batch_processor: Arc::new(BatchProcessor {
                batch_size: 1000,
                flush_interval: Duration::from_millis(100),
                max_retries: 3,
            }),
            validators: Arc::new(Vec::new()),
            transformers: Arc::new(Vec::new()),
        }
    }
    
    /// Ingest single feature set
    pub async fn ingest(&self, request: IngestionRequest) -> Result<(), FeatureStoreError> {
        // Validate features
        for (name, value) in &request.features {
            for validator in self.validators.iter() {
                validator.validate(name, value).await
                    .map_err(|e| FeatureStoreError::InvalidFeatureType)?;
            }
        }
        
        // Transform features
        let mut transformed_features = HashMap::new();
        for (name, value) in request.features {
            let transformed = self.transform_feature(&name, value).await?;
            transformed_features.insert(name, transformed);
        }
        
        // Queue for batch processing
        {
            let mut queue = self.queue.write();
            queue.push_back(IngestionRequest {
                entity_id: request.entity_id,
                entity_type: request.entity_type,
                features: request.features.clone(),
                timestamp: request.timestamp,
                metadata: request.metadata,
            });
        }
        
        // Process if batch is ready
        self.process_batch_if_ready().await?;
        
        Ok(())
    }
    
    /// Ingest from stream
    pub async fn ingest_stream<S>(&self, mut stream: S) -> Result<u64, FeatureStoreError>
    where
        S: futures::Stream<Item = IngestionRequest> + Unpin,
    {
        use futures::StreamExt;
        
        let mut count = 0u64;
        let mut batch = Vec::with_capacity(self.batch_processor.batch_size);
        
        while let Some(request) = stream.next().await {
            batch.push(request);
            count += 1;
            
            if batch.len() >= self.batch_processor.batch_size {
                self.write_batch(&batch).await?;
                batch.clear();
            }
        }
        
        // Write remaining
        if !batch.is_empty() {
            self.write_batch(&batch).await?;
        }
        
        Ok(count)
    }
    
    /// Ingest from Kafka/Kinesis
    pub async fn ingest_from_kafka(
        &self,
        topic: &str,
        consumer_group: &str,
    ) -> Result<(), FeatureStoreError> {
        use rdkafka::consumer::{StreamConsumer, Consumer};
        use rdkafka::config::ClientConfig;
        use rdkafka::message::Message;
        
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", consumer_group)
            .set("bootstrap.servers", "localhost:9092")
            .set("enable.auto.commit", "false")
            .create()
            .map_err(|e| FeatureStoreError::Serialization(
                serde_json::Error::custom(format!("Kafka error: {}", e))
            ))?;
        
        consumer.subscribe(&[topic])
            .map_err(|e| FeatureStoreError::Serialization(
                serde_json::Error::custom(format!("Subscribe error: {}", e))
            ))?;
        
        let mut batch = Vec::with_capacity(self.batch_processor.batch_size);
        
        loop {
            match consumer.recv().await {
                Ok(msg) => {
                    if let Some(payload) = msg.payload() {
                        match serde_json::from_slice::<IngestionRequest>(payload) {
                            Ok(request) => {
                                batch.push(request);
                                
                                if batch.len() >= self.batch_processor.batch_size {
                                    self.write_batch(&batch).await?;
                                    batch.clear();
                                    consumer.commit_consumer_state(rdkafka::consumer::CommitMode::Async)
                                        .map_err(|e| FeatureStoreError::Serialization(
                                            serde_json::Error::custom(format!("Commit error: {}", e))
                                        ))?;
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to parse message: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Kafka error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
    
    /// Compute and ingest derived features
    pub async fn compute_derived_features(
        &self,
        entity_id: &str,
        base_features: &HashMap<String, FeatureValue>,
    ) -> Result<HashMap<String, FeatureValue>, FeatureStoreError> {
        let mut derived = HashMap::new();
        
        // Moving averages
        if let Some(FeatureValue::Float(price)) = base_features.get("price") {
            let ma_7 = self.compute_moving_average(entity_id, "price", 7).await?;
            let ma_30 = self.compute_moving_average(entity_id, "price", 30).await?;
            
            derived.insert("price_ma_7".to_string(), FeatureValue::Float(ma_7));
            derived.insert("price_ma_30".to_string(), FeatureValue::Float(ma_30));
            
            // Price momentum
            let momentum = (price - ma_30) / ma_30;
            derived.insert("price_momentum".to_string(), FeatureValue::Float(momentum));
        }
        
        // Volatility features
        if let Some(FeatureValue::Float(volume)) = base_features.get("volume") {
            let vol_std = self.compute_volatility(entity_id, "volume", 24).await?;
            derived.insert("volume_volatility_24h".to_string(), FeatureValue::Float(vol_std));
        }
        
        // Technical indicators
        if let (Some(FeatureValue::Float(high)), Some(FeatureValue::Float(low)), Some(FeatureValue::Float(close))) = 
            (base_features.get("high"), base_features.get("low"), base_features.get("close")) {
            
            // RSI
            let rsi = self.compute_rsi(entity_id, 14).await?;
            derived.insert("rsi_14".to_string(), FeatureValue::Float(rsi));
            
            // ATR
            let atr = (high - low).abs();
            derived.insert("atr".to_string(), FeatureValue::Float(atr));
        }
        
        // Cross-entity features
        let market_features = self.compute_market_features(entity_id).await?;
        derived.extend(market_features);
        
        // Write derived features
        let timestamp = Utc::now();
        for (name, value) in &derived {
            self.write_single_feature(entity_id, name, value, timestamp).await?;
        }
        
        Ok(derived)
    }
    
    /// Transform raw feature to typed feature
    async fn transform_feature(&self, name: &str, raw: RawFeatureValue) -> Result<FeatureValue, FeatureStoreError> {
        // Apply custom transformers
        for transformer in self.transformers.iter() {
            match transformer.transform(name, raw.clone()).await {
                Ok(transformed) => return Ok(transformed),
                Err(TransformError::Unsupported) => continue,
                Err(e) => return Err(FeatureStoreError::InvalidFeatureType),
            }
        }
        
        // Default transformation
        Ok(match raw {
            RawFeatureValue::Float(v) => FeatureValue::Float(v),
            RawFeatureValue::Integer(v) => FeatureValue::Integer(v),
            RawFeatureValue::String(v) => FeatureValue::String(v),
            RawFeatureValue::Boolean(v) => FeatureValue::Boolean(v),
            RawFeatureValue::Json(v) => FeatureValue::Json(v),
            RawFeatureValue::Binary(v) => {
                // Convert binary to base64 string
                FeatureValue::String(base64::encode(v))
            }
        })
    }
    
    /// Process batch if ready
    async fn process_batch_if_ready(&self) -> Result<(), FeatureStoreError> {
        let batch = {
            let mut queue = self.queue.write();
            if queue.len() >= self.batch_processor.batch_size {
                let batch: Vec<_> = queue.drain(..self.batch_processor.batch_size).collect();
                Some(batch)
            } else {
                None
            }
        };
        
        if let Some(batch) = batch {
            self.write_batch(&batch).await?;
        }
        
        Ok(())
    }
    
    /// Write batch to database
    async fn write_batch(&self, batch: &[IngestionRequest]) -> Result<(), FeatureStoreError> {
        let start = Instant::now();
        
        // Prepare batch insert
        let mut query = String::from(
            "INSERT INTO features (feature_name, entity_id, entity_type, feature_value, timestamp, metadata) VALUES "
        );
        
        let mut values = Vec::new();
        for (i, request) in batch.iter().enumerate() {
            for (name, value) in &request.features {
                if i > 0 || values.len() > 0 {
                    query.push_str(", ");
                }
                
                // Transform raw value
                let feature_value = self.transform_feature(name, value.clone()).await?;
                
                query.push_str(&format!(
                    "(${}::text, ${}::text, ${}::text, ${}::jsonb, ${}::timestamptz, ${}::jsonb)",
                    values.len() + 1,
                    values.len() + 2,
                    values.len() + 3,
                    values.len() + 4,
                    values.len() + 5,
                    values.len() + 6,
                ));
                
                values.push(name.clone());
                values.push(request.entity_id.clone());
                values.push(request.entity_type.clone());
                values.push(serde_json::to_value(&feature_value)?);
                values.push(request.timestamp);
                values.push(request.metadata.clone().unwrap_or(serde_json::Value::Null));
            }
        }
        
        query.push_str(" ON CONFLICT (feature_name, entity_id, timestamp, version) DO NOTHING");
        
        // Execute with retries
        let mut retries = 0;
        loop {
            match sqlx::query(&query)
                .execute(self.store.pool.as_ref())
                .await
            {
                Ok(_) => {
                    let latency = start.elapsed();
                    let mut metrics = self.store.metrics.write();
                    metrics.total_writes += batch.len() as u64;
                    metrics.avg_write_latency_us = latency.as_micros() as f64;
                    break;
                }
                Err(e) if retries < self.batch_processor.max_retries => {
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(100 * retries as u64)).await;
                }
                Err(e) => return Err(FeatureStoreError::Database(e)),
            }
        }
        
        Ok(())
    }
    
    /// Write single feature
    async fn write_single_feature(
        &self,
        entity_id: &str,
        feature_name: &str,
        value: &FeatureValue,
        timestamp: DateTime<Utc>,
    ) -> Result<(), FeatureStoreError> {
        sqlx::query(r#"
            INSERT INTO features (feature_name, entity_id, entity_type, feature_value, timestamp)
            VALUES ($1, $2, 'derived', $3, $4)
            ON CONFLICT (feature_name, entity_id, timestamp, version) DO NOTHING
        "#)
        .bind(feature_name)
        .bind(entity_id)
        .bind(serde_json::to_value(value)?)
        .bind(timestamp)
        .execute(self.store.pool.as_ref())
        .await?;
        
        Ok(())
    }
    
    /// Compute moving average
    async fn compute_moving_average(
        &self,
        entity_id: &str,
        feature_name: &str,
        window_size: i32,
    ) -> Result<f64, FeatureStoreError> {
        let result = sqlx::query_as::<_, (Option<f64>,)>(r#"
            SELECT AVG((feature_value->>'value')::float)
            FROM (
                SELECT feature_value
                FROM features
                WHERE entity_id = $1 AND feature_name = $2
                ORDER BY timestamp DESC
                LIMIT $3
            ) t
        "#)
        .bind(entity_id)
        .bind(feature_name)
        .bind(window_size)
        .fetch_one(self.store.pool.as_ref())
        .await?;
        
        Ok(result.0.unwrap_or(0.0))
    }
    
    /// Compute volatility
    async fn compute_volatility(
        &self,
        entity_id: &str,
        feature_name: &str,
        window_hours: i32,
    ) -> Result<f64, FeatureStoreError> {
        let result = sqlx::query_as::<_, (Option<f64>,)>(r#"
            SELECT STDDEV((feature_value->>'value')::float)
            FROM features
            WHERE entity_id = $1 
                AND feature_name = $2
                AND timestamp > NOW() - INTERVAL '$3 hours'
        "#)
        .bind(entity_id)
        .bind(feature_name)
        .bind(window_hours)
        .fetch_one(self.store.pool.as_ref())
        .await?;
        
        Ok(result.0.unwrap_or(0.0))
    }
    
    /// Compute RSI
    async fn compute_rsi(&self, entity_id: &str, period: i32) -> Result<f64, FeatureStoreError> {
        // Simplified RSI calculation
        let prices = sqlx::query_as::<_, (f64,)>(r#"
            SELECT (feature_value->>'value')::float
            FROM features
            WHERE entity_id = $1 AND feature_name = 'close'
            ORDER BY timestamp DESC
            LIMIT $2
        "#)
        .bind(entity_id)
        .bind(period + 1)
        .fetch_all(self.store.pool.as_ref())
        .await?;
        
        if prices.len() < 2 {
            return Ok(50.0); // Neutral RSI
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for window in prices.windows(2) {
            let change = window[0].0 - window[1].0;
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            Ok(100.0)
        } else {
            let rs = avg_gain / avg_loss;
            Ok(100.0 - (100.0 / (1.0 + rs)))
        }
    }
    
    /// Compute market-wide features
    async fn compute_market_features(&self, entity_id: &str) -> Result<HashMap<String, FeatureValue>, FeatureStoreError> {
        let mut features = HashMap::new();
        
        // Market correlation
        let correlation = sqlx::query_as::<_, (Option<f64>,)>(r#"
            SELECT CORR(a.val, b.val)
            FROM (
                SELECT (feature_value->>'value')::float as val, timestamp
                FROM features
                WHERE entity_id = $1 AND feature_name = 'price'
            ) a
            JOIN (
                SELECT (feature_value->>'value')::float as val, timestamp
                FROM features
                WHERE entity_id = 'BTC' AND feature_name = 'price'
            ) b ON a.timestamp = b.timestamp
        "#)
        .bind(entity_id)
        .fetch_one(self.store.pool.as_ref())
        .await?;
        
        features.insert("market_correlation".to_string(), 
            FeatureValue::Float(correlation.0.unwrap_or(0.0)));
        
        Ok(features)
    }
}

use base64;
use rdkafka;

// Drew: "This ingestion pipeline can handle millions of features per second!"