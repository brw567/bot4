//! # FEATURE SCHEMA - Type safety and validation
//! Quinn (Safety Lead): "Type safety prevents feature drift"

use super::*;
use serde_json::Value;

/// Feature schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub name: String,
    pub description: String,
    pub data_type: DataType,
    pub constraints: Vec<Constraint>,
    pub default_value: Option<FeatureValue>,
    pub is_required: bool,
    pub is_derived: bool,
    pub derivation_function: Option<String>,
}

/// Data type definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Float { min: Option<f64>, max: Option<f64>, precision: Option<u8> },
    Integer { min: Option<i64>, max: Option<i64> },
    String { max_length: Option<usize>, pattern: Option<String> },
    Boolean,
    Timestamp,
    Vector { dimension: usize, element_type: Box<DataType> },
    Categorical { values: Vec<String> },
    Json { schema: Option<Value> },
}

/// Constraint definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    NotNull,
    Unique,
    Range { min: f64, max: f64 },
    Regex { pattern: String },
    Custom { validator: String },
    ForeignKey { entity: String, field: String },
}

/// Schema registry
pub struct SchemaRegistry {
    schemas: HashMap<String, FeatureSchema>,
    entity_schemas: HashMap<String, EntitySchema>,
}

/// Entity schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySchema {
    pub entity_type: String,
    pub id_field: String,
    pub features: Vec<String>,
    pub required_features: Vec<String>,
    pub composite_keys: Vec<Vec<String>>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            schemas: HashMap::new(),
            entity_schemas: HashMap::new(),
        };
        
        // Register default schemas
        registry.register_default_schemas();
        registry
    }
    
    fn register_default_schemas(&mut self) {
        // Price features
        self.schemas.insert("price".to_string(), FeatureSchema {
            name: "price".to_string(),
            description: "Current asset price".to_string(),
            data_type: DataType::Float { min: Some(0.0), max: None, precision: Some(8) },
            constraints: vec![Constraint::NotNull, Constraint::Range { min: 0.0, max: 1_000_000.0 }],
            default_value: None,
            is_required: true,
            is_derived: false,
            derivation_function: None,
        });
        
        // Volume features
        self.schemas.insert("volume".to_string(), FeatureSchema {
            name: "volume".to_string(),
            description: "Trading volume".to_string(),
            data_type: DataType::Float { min: Some(0.0), max: None, precision: Some(8) },
            constraints: vec![Constraint::NotNull, Constraint::Range { min: 0.0, max: f64::MAX }],
            default_value: Some(FeatureValue::Float(0.0)),
            is_required: true,
            is_derived: false,
            derivation_function: None,
        });
        
        // Technical indicators
        self.schemas.insert("rsi_14".to_string(), FeatureSchema {
            name: "rsi_14".to_string(),
            description: "14-period RSI".to_string(),
            data_type: DataType::Float { min: Some(0.0), max: Some(100.0), precision: Some(2) },
            constraints: vec![Constraint::Range { min: 0.0, max: 100.0 }],
            default_value: Some(FeatureValue::Float(50.0)),
            is_required: false,
            is_derived: true,
            derivation_function: Some("compute_rsi".to_string()),
        });
        
        // Market microstructure
        self.schemas.insert("bid_ask_spread".to_string(), FeatureSchema {
            name: "bid_ask_spread".to_string(),
            description: "Current bid-ask spread".to_string(),
            data_type: DataType::Float { min: Some(0.0), max: None, precision: Some(8) },
            constraints: vec![Constraint::Range { min: 0.0, max: 100.0 }],
            default_value: None,
            is_required: false,
            is_derived: false,
            derivation_function: None,
        });
        
        // Vector features for embeddings
        self.schemas.insert("price_embedding".to_string(), FeatureSchema {
            name: "price_embedding".to_string(),
            description: "Price pattern embedding".to_string(),
            data_type: DataType::Vector { 
                dimension: 128, 
                element_type: Box::new(DataType::Float { min: Some(-1.0), max: Some(1.0), precision: None })
            },
            constraints: vec![],
            default_value: None,
            is_required: false,
            is_derived: true,
            derivation_function: Some("generate_price_embedding".to_string()),
        });
    }
    
    /// Validate feature value against schema
    pub fn validate(&self, feature_name: &str, value: &FeatureValue) -> Result<(), ValidationError> {
        let schema = self.schemas.get(feature_name)
            .ok_or_else(|| ValidationError::SchemaViolation(format!("Unknown feature: {}", feature_name)))?;
        
        // Type validation
        match (&schema.data_type, value) {
            (DataType::Float { min, max, .. }, FeatureValue::Float(v)) => {
                if let Some(min_val) = min {
                    if v < min_val {
                        return Err(ValidationError::OutOfRange(format!("{} < {}", v, min_val)));
                    }
                }
                if let Some(max_val) = max {
                    if v > max_val {
                        return Err(ValidationError::OutOfRange(format!("{} > {}", v, max_val)));
                    }
                }
            }
            (DataType::Integer { min, max }, FeatureValue::Integer(v)) => {
                if let Some(min_val) = min {
                    if v < min_val {
                        return Err(ValidationError::OutOfRange(format!("{} < {}", v, min_val)));
                    }
                }
                if let Some(max_val) = max {
                    if v > max_val {
                        return Err(ValidationError::OutOfRange(format!("{} > {}", v, max_val)));
                    }
                }
            }
            (DataType::String { max_length, pattern }, FeatureValue::String(s)) => {
                if let Some(max_len) = max_length {
                    if s.len() > *max_len {
                        return Err(ValidationError::SchemaViolation(format!("String too long: {} > {}", s.len(), max_len)));
                    }
                }
                if let Some(pattern_str) = pattern {
                    let re = regex::Regex::new(pattern_str)
                        .map_err(|e| ValidationError::SchemaViolation(format!("Invalid regex: {}", e)))?;
                    if !re.is_match(s) {
                        return Err(ValidationError::SchemaViolation(format!("String doesn't match pattern: {}", pattern_str)));
                    }
                }
            }
            (DataType::Boolean, FeatureValue::Boolean(_)) => {}
            (DataType::Vector { dimension, .. }, FeatureValue::Vector(v)) => {
                if v.len() != *dimension {
                    return Err(ValidationError::SchemaViolation(
                        format!("Vector dimension mismatch: {} != {}", v.len(), dimension)
                    ));
                }
            }
            (DataType::Categorical { values }, FeatureValue::String(s)) => {
                if !values.contains(s) {
                    return Err(ValidationError::SchemaViolation(
                        format!("Invalid categorical value: {}", s)
                    ));
                }
            }
            _ => {
                return Err(ValidationError::InvalidType(format!(
                    "Type mismatch for feature: {}", feature_name
                )));
            }
        }
        
        // Constraint validation
        for constraint in &schema.constraints {
            match constraint {
                Constraint::NotNull => {
                    // Already validated by type system
                }
                Constraint::Range { min, max } => {
                    if let FeatureValue::Float(v) = value {
                        if v < min || v > max {
                            return Err(ValidationError::OutOfRange(
                                format!("{} not in range [{}, {}]", v, min, max)
                            ));
                        }
                    }
                }
                Constraint::Regex { pattern } => {
                    if let FeatureValue::String(s) = value {
                        let re = regex::Regex::new(pattern)
                            .map_err(|e| ValidationError::SchemaViolation(format!("Invalid regex: {}", e)))?;
                        if !re.is_match(s) {
                            return Err(ValidationError::SchemaViolation(
                                format!("Value doesn't match regex: {}", pattern)
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }
}

use regex;

// Quinn: "Schema validation prevents 99% of feature engineering bugs"