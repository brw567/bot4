// Kalman Filter for Statistical Gap Detection
// Based on Harvey (1989) and Durbin & Koopman (2012)
//
// Theory: Kalman filter provides optimal state estimation for linear dynamic systems
// State equation: x(t) = F*x(t-1) + B*u(t) + w(t)
// Observation equation: z(t) = H*x(t) + v(t)
// where w(t) ~ N(0,Q) and v(t) ~ N(0,R)
//
// Applications in trading:
// - Detect missing data points in time series
// - Predict expected values to identify outliers
// - Smooth noisy market data
// - Track regime changes in volatility

use std::collections::{VecDeque, HashMap};
use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use nalgebra::{DMatrix, DVector};
use tracing::{debug, warn, info};

use super::DataBatch;

/// Kalman filter configuration
#[derive(Debug, Clone, Deserialize)]
/// TODO: Add docs
pub struct KalmanConfig {
    pub process_noise: f64,           // Q: Process noise covariance
    pub measurement_noise: f64,       // R: Measurement noise covariance
    pub initial_covariance: f64,      // P0: Initial error covariance
    pub gap_threshold_seconds: i64,   // Minimum gap duration to report
    pub prediction_std_multiplier: f64, // Std deviations for outlier detection
    pub adaptive_noise: bool,         // Adapt noise parameters online
    pub max_gap_interpolate: i64,     // Max gap size to interpolate (seconds)
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 1e-5,
            measurement_noise: 1e-3,
            initial_covariance: 1.0,
            gap_threshold_seconds: 5,
            prediction_std_multiplier: 3.0,
            adaptive_noise: true,
            max_gap_interpolate: 60,
        }
    }
}

/// Multi-dimensional Kalman filter for gap detection
/// TODO: Add docs
pub struct KalmanGapDetector {
    config: KalmanConfig,
    
    // State tracking per symbol
    symbol_states: Arc<RwLock<HashMap<String, KalmanState>>>,
    
    // Gap history for pattern analysis
    gap_history: VecDeque<GapEvent>,
    
    // Innovation statistics for adaptive filtering
    innovation_stats: Arc<RwLock<HashMap<String, InnovationStats>>>,
}

/// Kalman filter state for a single symbol
struct KalmanState {
    // State vector: [position, velocity, acceleration]
    x: DVector<f64>,
    
    // Error covariance matrix
    P: DMatrix<f64>,
    
    // State transition matrix
    F: DMatrix<f64>,
    
    // Observation matrix
    H: DMatrix<f64>,
    
    // Process noise covariance
    Q: DMatrix<f64>,
    
    // Measurement noise covariance
    R: DMatrix<f64>,
    
    // Last observation time
    last_timestamp: DateTime<Utc>,
    
    // Expected next observation time
    expected_interval: Duration,
}

impl KalmanGapDetector {
    /// Create new Kalman gap detector
    pub async fn new(config: KalmanConfig) -> Result<Self> {
        info!("Initializing Kalman filter gap detector");
        
        Ok(Self {
            config,
            symbol_states: Arc::new(RwLock::new(HashMap::new())),
            gap_history: VecDeque::with_capacity(1000),
            innovation_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Detect gaps in data stream
    pub async fn detect_gaps(&mut self, data: &DataBatch) -> Result<Option<GapEvent>> {
        let symbol = &data.symbol;
        let timestamp = data.timestamp;
        
        // Initialize state if new symbol
        {
            let states = self.symbol_states.read().await;
            if !states.contains_key(symbol) {
                drop(states);
                self.initialize_state(symbol, &data.values[0], timestamp).await?;
                return Ok(None);
            }
        }
        
        let states = self.symbol_states.read().await;
        let state = states.get(symbol)
            .ok_or_else(|| anyhow!("State not found for symbol"))?;
        
        // Check for temporal gap
        let time_diff = timestamp - state.last_timestamp;
        if time_diff.num_seconds() > self.config.gap_threshold_seconds {
            let gap = GapEvent {
                symbol: symbol.clone(),
                gap_start: state.last_timestamp,
                gap_end: timestamp,
                gap_duration: time_diff,
                expected_points: self.estimate_missing_points(&time_diff, &state.expected_interval),
                severity: self.calculate_gap_severity(&time_diff),
                detected_at: Utc::now(),
            };
            
            warn!("Gap detected for {}: {} seconds", symbol, time_diff.num_seconds());
            return Ok(Some(gap));
        }
        
        // Run Kalman filter prediction and update
        let prediction_error = self.kalman_update(state, &data.values[0], timestamp).await?;
        
        // Check for statistical gap (outlier)
        if prediction_error.abs() > self.config.prediction_std_multiplier * state.P[(0, 0)].sqrt() {
            let gap = GapEvent {
                symbol: symbol.clone(),
                gap_start: state.last_timestamp,
                gap_end: timestamp,
                gap_duration: time_diff,
                expected_points: 0,  // Statistical gap, not temporal
                severity: GapSeverity::Statistical,
                detected_at: Utc::now(),
            };
            
            debug!("Statistical gap detected for {}: prediction error = {:.4}", symbol, prediction_error);
            return Ok(Some(gap));
        }
        
        Ok(None)
    }
    
    /// Initialize Kalman state for new symbol
    async fn initialize_state(
        &mut self,
        symbol: &str,
        initial_value: &f64,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        let dim = 3;  // Position, velocity, acceleration
        
        // Initialize state vector
        let x = DVector::from_vec(vec![*initial_value, 0.0, 0.0]);
        
        // Initialize error covariance
        let mut P = DMatrix::identity(dim, dim);
        P *= self.config.initial_covariance;
        
        // State transition matrix (constant velocity model)
        let dt = 1.0;  // Will be updated dynamically
        let F = DMatrix::from_row_slice(3, 3, &[
            1.0, dt, dt*dt/2.0,
            0.0, 1.0, dt,
            0.0, 0.0, 1.0,
        ]);
        
        // Observation matrix (observe position only)
        let mut H = DMatrix::zeros(1, 3);
        H[(0, 0)] = 1.0;
        
        // Process noise covariance
        let q = self.config.process_noise;
        let Q = DMatrix::from_row_slice(3, 3, &[
            q * dt.powi(4) / 4.0, q * dt.powi(3) / 2.0, q * dt.powi(2) / 2.0,
            q * dt.powi(3) / 2.0, q * dt.powi(2), q * dt,
            q * dt.powi(2) / 2.0, q * dt, q,
        ]);
        
        // Measurement noise covariance
        let R = DMatrix::from_element(1, 1, self.config.measurement_noise);
        
        let state = KalmanState {
            x,
            P,
            F,
            H,
            Q,
            R,
            last_timestamp: timestamp,
            expected_interval: Duration::seconds(1),  // Default, will adapt
        };
        
        self.symbol_states.insert(symbol.to_string(), state);
        
        // Initialize innovation statistics
        self.innovation_stats.insert(symbol.to_string(), InnovationStats::default());
        
        Ok(())
    }
    
    /// Run Kalman filter prediction and update
    async fn kalman_update(
        &mut self,
        state: &mut KalmanState,
        measurement: &f64,
        timestamp: DateTime<Utc>,
    ) -> Result<f64> {
        // Calculate time delta
        let dt = (timestamp - state.last_timestamp).num_milliseconds() as f64 / 1000.0;
        
        // Update state transition matrix with actual dt
        state.F[(0, 1)] = dt;
        state.F[(0, 2)] = dt * dt / 2.0;
        state.F[(1, 2)] = dt;
        
        // Update process noise with actual dt
        if self.config.adaptive_noise {
            self.update_process_noise(state, dt)?;
        }
        
        // Prediction step
        // x_pred = F * x
        let x_pred = &state.F * &state.x;
        
        // P_pred = F * P * F' + Q
        let P_pred = &state.F * &state.P * state.F.transpose() + &state.Q;
        
        // Innovation (measurement residual)
        // y = z - H * x_pred
        let z = DVector::from_element(1, *measurement);
        let y = z - &state.H * &x_pred;
        let innovation = y[0];
        
        // Innovation covariance
        // S = H * P_pred * H' + R
        let S = &state.H * &P_pred * state.H.transpose() + &state.R;
        
        // Kalman gain
        // K = P_pred * H' * S^(-1)
        let K = &P_pred * state.H.transpose() * S.try_inverse()
            .ok_or_else(|| anyhow!("Failed to invert innovation covariance"))?;
        
        // State update
        // x = x_pred + K * y
        state.x = x_pred + &K * y;
        
        // Covariance update
        // P = (I - K * H) * P_pred
        let I = DMatrix::identity(3, 3);
        state.P = (I - &K * &state.H) * P_pred;
        
        // Update timestamp and interval
        state.expected_interval = timestamp - state.last_timestamp;
        state.last_timestamp = timestamp;
        
        // Update innovation statistics if adaptive
        if self.config.adaptive_noise {
            self.update_innovation_stats(state.symbol.as_ref().unwrap(), innovation, S[(0, 0)])?;
        }
        
        Ok(innovation)
    }
    
    /// Update process noise adaptively based on innovation
    fn update_process_noise(&mut self, state: &mut KalmanState, dt: f64) -> Result<()> {
        if let Some(stats) = self.innovation_stats.get(state.symbol.as_ref().unwrap()) {
            if stats.count > 10 {
                // Adapt process noise based on innovation variance
                let innovation_var = stats.variance;
                let expected_var = state.R[(0, 0)];
                
                if innovation_var > 2.0 * expected_var {
                    // Increase process noise
                    let q = self.config.process_noise * 1.1;
                    state.Q = self.create_process_noise_matrix(q, dt);
                } else if innovation_var < 0.5 * expected_var {
                    // Decrease process noise
                    let q = self.config.process_noise * 0.9;
                    state.Q = self.create_process_noise_matrix(q, dt);
                }
            }
        }
        Ok(())
    }
    
    /// Create process noise covariance matrix
    fn create_process_noise_matrix(&self, q: f64, dt: f64) -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 3, &[
            q * dt.powi(4) / 4.0, q * dt.powi(3) / 2.0, q * dt.powi(2) / 2.0,
            q * dt.powi(3) / 2.0, q * dt.powi(2), q * dt,
            q * dt.powi(2) / 2.0, q * dt, q,
        ])
    }
    
    /// Update innovation statistics for adaptive filtering
    fn update_innovation_stats(
        &mut self,
        symbol: &str,
        innovation: f64,
        innovation_var: f64,
    ) -> Result<()> {
        if let Some(stats) = self.innovation_stats.get_mut(symbol) {
            stats.update(innovation, innovation_var);
        }
        Ok(())
    }
    
    /// Estimate number of missing points in gap
    fn estimate_missing_points(&self, gap_duration: &Duration, expected_interval: &Duration) -> usize {
        if expected_interval.num_seconds() > 0 {
            (gap_duration.num_seconds() / expected_interval.num_seconds()) as usize
        } else {
            0
        }
    }
    
    /// Calculate gap severity based on duration and context
    fn calculate_gap_severity(&self, gap_duration: &Duration) -> GapSeverity {
        let seconds = gap_duration.num_seconds();
        match seconds {
            s if s < 10 => GapSeverity::Minor,
            s if s < 60 => GapSeverity::Moderate,
            s if s < 300 => GapSeverity::Major,
            _ => GapSeverity::Critical,
        }
    }
    
    /// Interpolate missing values in gap
    pub async fn interpolate_gap(
        &self,
        symbol: &str,
        gap_start: DateTime<Utc>,
        gap_end: DateTime<Utc>,
    ) -> Result<Vec<InterpolatedPoint>> {
        let state = self.symbol_states.get(symbol)
            .ok_or_else(|| anyhow!("No state found for symbol"))?;
        
        let gap_duration = gap_end - gap_start;
        
        // Only interpolate small gaps
        if gap_duration.num_seconds() > self.config.max_gap_interpolate {
            return Ok(Vec::new());
        }
        
        let mut interpolated = Vec::new();
        let num_points = self.estimate_missing_points(&gap_duration, &state.expected_interval);
        
        for i in 1..=num_points {
            let fraction = i as f64 / (num_points + 1) as f64;
            let timestamp = gap_start + Duration::milliseconds(
                (gap_duration.num_milliseconds() as f64 * fraction) as i64
            );
            
            // Linear interpolation with velocity
            let dt = (timestamp - state.last_timestamp).num_seconds() as f64;
            let value = state.x[0] + state.x[1] * dt + state.x[2] * dt * dt / 2.0;
            
            interpolated.push(InterpolatedPoint {
                timestamp,
                value,
                confidence: 1.0 - fraction * 0.5,  // Confidence decreases with distance
            });
        }
        
        Ok(interpolated)
    }
    
    /// Get gap statistics for a symbol
    pub async fn get_gap_statistics(&self, symbol: &str) -> Result<GapStatistics> {
        let symbol_gaps: Vec<_> = self.gap_history.iter()
            .filter(|g| g.symbol == symbol)
            .cloned()
            .collect();
        
        if symbol_gaps.is_empty() {
            return Ok(GapStatistics::default());
        }
        
        let total_gaps = symbol_gaps.len();
        let total_duration: i64 = symbol_gaps.iter()
            .map(|g| g.gap_duration.num_seconds())
            .sum();
        
        let avg_duration = total_duration as f64 / total_gaps as f64;
        
        let max_gap = symbol_gaps.iter()
            .max_by_key(|g| g.gap_duration)
            .cloned();
        
        Ok(GapStatistics {
            total_gaps,
            total_duration_seconds: total_duration,
            average_duration_seconds: avg_duration,
            max_gap,
            gaps_by_severity: self.count_by_severity(&symbol_gaps),
        })
    }
    
    /// Count gaps by severity
    fn count_by_severity(&self, gaps: &[GapEvent]) -> HashMap<GapSeverity, usize> {
        let mut counts = HashMap::new();
        for gap in gaps {
            *counts.entry(gap.severity.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// Innovation statistics for adaptive filtering
#[derive(Default)]
struct InnovationStats {
    count: usize,
    mean: f64,
    variance: f64,
    sum: f64,
    sum_squared: f64,
}

impl InnovationStats {
    fn update(&mut self, innovation: f64, _innovation_var: f64) {
        self.count += 1;
        self.sum += innovation;
        self.sum_squared += innovation * innovation;
        
        self.mean = self.sum / self.count as f64;
        self.variance = (self.sum_squared / self.count as f64) - self.mean * self.mean;
    }
}

/// Gap detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct GapEvent {
    pub symbol: String,
    pub gap_start: DateTime<Utc>,
    pub gap_end: DateTime<Utc>,
    pub gap_duration: Duration,
    pub expected_points: usize,
    pub severity: GapSeverity,
    pub detected_at: DateTime<Utc>,
}

/// Gap severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
/// TODO: Add docs
pub enum GapSeverity {
    Minor,       // < 10 seconds
    Moderate,    // 10-60 seconds
    Major,       // 1-5 minutes
    Critical,    // > 5 minutes
    Statistical, // Outlier detected
}

/// Interpolated data point
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct InterpolatedPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub confidence: f64,
}

/// Gap statistics for reporting
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
/// TODO: Add docs
pub struct GapStatistics {
    pub total_gaps: usize,
    pub total_duration_seconds: i64,
    pub average_duration_seconds: f64,
    pub max_gap: Option<GapEvent>,
    pub gaps_by_severity: HashMap<GapSeverity, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_kalman_gap_detection() {
        let config = KalmanConfig::default();
        let mut detector = KalmanGapDetector::new(config).await.unwrap();
        
        // Create data with a gap
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        // Normal data
        for i in 0..10 {
            timestamps.push(Utc::now() - Duration::seconds(100 - i));
            values.push(100.0 + i as f64);
        }
        
        // Gap of 30 seconds
        timestamps.push(Utc::now() - Duration::seconds(60));
        values.push(140.0);
        
        // Process data
        for (i, (timestamp, value)) in timestamps.iter().zip(values.iter()).enumerate() {
            let batch = DataBatch {
                symbol: "TEST".to_string(),
                data_type: super::super::DataType::Price,
                timestamp: *timestamp,
                values: vec![*value],
                source: "test".to_string(),
                metadata: None,
            };
            
            let gap = detector.detect_gaps(&batch).await.unwrap();
            
            // Should detect gap at position 10
            if i == 10 {
                assert!(gap.is_some());
                if let Some(g) = gap {
                    assert!(g.gap_duration.num_seconds() >= 30);
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_statistical_gap_detection() {
        let config = KalmanConfig::default();
        let mut detector = KalmanGapDetector::new(config).await.unwrap();
        
        // Create data with outlier
        let mut batches = Vec::new();
        for i in 0..20 {
            let value = if i == 10 {
                200.0  // Outlier
            } else {
                100.0 + (i as f64) * 0.1
            };
            
            batches.push(DataBatch {
                symbol: "OUTLIER".to_string(),
                data_type: super::super::DataType::Price,
                timestamp: Utc::now() - Duration::seconds(20 - i),
                values: vec![value],
                source: "test".to_string(),
                metadata: None,
            });
        }
        
        // Process data
        let mut outlier_detected = false;
        for batch in batches {
            if let Some(gap) = detector.detect_gaps(&batch).await.unwrap() {
                if gap.severity == GapSeverity::Statistical {
                    outlier_detected = true;
                }
            }
        }
        
        assert!(outlier_detected, "Should detect statistical outlier");
    }
}