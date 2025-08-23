// Auto-Tuning Database Persistence Layer
// Team: FULL TEAM DEEP DIVE - NO SIMPLIFICATIONS!
// Alex: "Every parameter MUST be saved and restored from DB!"
// Quinn: "We can't lose our learned optimizations on restart!"

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sqlx::{PgPool, postgres::PgPoolOptions};
use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};

/// Database persistence for auto-tuning parameters
/// Sam: "This is CRITICAL - we need ACID compliance!"
pub struct AutoTuningPersistence {
    pool: PgPool,
}

/// Adaptive parameter stored in database
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AdaptiveParameter {
    pub id: uuid::Uuid,
    pub parameter_name: String,
    pub current_value: Decimal,
    pub min_value: Decimal,
    pub max_value: Decimal,
    pub default_value: Decimal,
    pub last_adjustment: DateTime<Utc>,
    pub adjustment_count: i32,
    pub adjustment_reason: Option<String>,
    pub performance_impact: Option<Decimal>,
    pub stability_score: Option<Decimal>,
    pub optimal_for_regime: Option<String>,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub category: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Q-Learning state-action-value entry
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct QLearningEntry {
    pub id: uuid::Uuid,
    pub state_hash: String,
    pub market_regime: String,
    pub volatility_bucket: i32,
    pub drawdown_bucket: i32,
    pub momentum_bucket: i32,
    pub action_id: i32,
    pub action_description: Option<String>,
    pub q_value: Decimal,
    pub visit_count: i32,
    pub total_reward: Decimal,
    pub avg_reward: Decimal,
    pub learning_rate: Decimal,
    pub discount_factor: Decimal,
    pub exploration_rate: Decimal,
    pub first_seen: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// Market regime history entry
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct MarketRegimeEntry {
    pub id: uuid::Uuid,
    pub timestamp: DateTime<Utc>,
    pub regime: String,
    pub confidence: Decimal,
    pub trend_slope: Option<Decimal>,
    pub average_return: Option<Decimal>,
    pub volatility: Option<Decimal>,
    pub volume_surge: bool,
    pub correlation_stable: bool,
    pub vix_level: Option<Decimal>,
    pub volume_24h: Option<Decimal>,
    pub bid_ask_spread: Option<Decimal>,
    pub previous_regime: Option<String>,
    pub regime_duration_hours: Option<i32>,
    pub created_at: DateTime<Utc>,
}

/// Performance feedback for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PerformanceFeedback {
    pub id: uuid::Uuid,
    pub timestamp: DateTime<Utc>,
    pub market_regime: String,
    pub position_id: Option<uuid::Uuid>,
    pub signal_id: Option<String>,
    pub action_type: Option<String>,
    pub old_value: Option<Decimal>,
    pub new_value: Option<Decimal>,
    pub pnl: Option<Decimal>,
    pub return_pct: Option<Decimal>,
    pub sharpe_contribution: Option<Decimal>,
    pub max_drawdown: Option<Decimal>,
    pub immediate_reward: Option<Decimal>,
    pub delayed_reward: Option<Decimal>,
    pub total_reward: Option<Decimal>,
    pub success: bool,
    pub success_criteria: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl AutoTuningPersistence {
    /// Create new persistence layer
    /// Avery: "Connection pooling for optimal performance!"
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .min_connections(2)
            .connect(database_url)
            .await?;
        
        Ok(Self { pool })
    }
    
    /// Load all adaptive parameters from database
    /// Quinn: "We need these on startup to maintain our edge!"
    pub async fn load_adaptive_parameters(&self) -> Result<HashMap<String, AdaptiveParameter>> {
        let params = sqlx::query_as::<_, AdaptiveParameter>(
            "SELECT * FROM adaptive_parameters ORDER BY parameter_name"
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut map = HashMap::new();
        for param in params {
            map.insert(param.parameter_name.clone(), param);
        }
        
        Ok(map)
    }
    
    /// Update adaptive parameter with audit trail
    /// Alex: "EVERY change must be tracked!"
    pub async fn update_parameter(
        &self,
        name: &str,
        new_value: Decimal,
        reason: &str,
        regime: Option<&str>,
    ) -> Result<()> {
        // Start transaction for atomicity
        let mut tx = self.pool.begin().await?;
        
        // Get current value for audit
        let old_value: Option<Decimal> = sqlx::query_scalar(
            "SELECT current_value FROM adaptive_parameters WHERE parameter_name = $1"
        )
        .bind(name)
        .fetch_optional(&mut *tx)
        .await?;
        
        // Update parameter
        sqlx::query(
            "UPDATE adaptive_parameters 
             SET current_value = $1, 
                 last_adjustment = NOW(),
                 adjustment_count = adjustment_count + 1,
                 adjustment_reason = $2,
                 updated_at = NOW()
             WHERE parameter_name = $3"
        )
        .bind(new_value)
        .bind(reason)
        .bind(name)
        .execute(&mut *tx)
        .await?;
        
        // Create audit entry
        if let Some(old) = old_value {
            let change_pct = ((new_value - old) / old) * dec!(100);
            
            sqlx::query(
                "INSERT INTO auto_tuning_audit 
                 (parameter_name, old_value, new_value, change_percentage, 
                  trigger_reason, market_regime, adjusted_by)
                 VALUES ($1, $2, $3, $4, $5, $6, 'auto_tuner')"
            )
            .bind(name)
            .bind(old)
            .bind(new_value)
            .bind(change_pct)
            .bind(reason)
            .bind(regime)
            .execute(&mut *tx)
            .await?;
        }
        
        // Commit transaction
        tx.commit().await?;
        
        Ok(())
    }
    
    /// Save Q-Learning table to database
    /// Morgan: "This is our learned intelligence - MUST persist!"
    pub async fn save_q_table(&self, entries: Vec<QLearningEntry>) -> Result<()> {
        for entry in entries {
            sqlx::query(
                "INSERT INTO q_learning_table 
                 (state_hash, market_regime, volatility_bucket, drawdown_bucket,
                  momentum_bucket, action_id, q_value, visit_count, total_reward,
                  avg_reward, learning_rate, discount_factor, exploration_rate)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                 ON CONFLICT (state_hash, action_id) 
                 DO UPDATE SET 
                    q_value = EXCLUDED.q_value,
                    visit_count = EXCLUDED.visit_count,
                    total_reward = EXCLUDED.total_reward,
                    avg_reward = EXCLUDED.avg_reward,
                    last_updated = NOW()"
            )
            .bind(&entry.state_hash)
            .bind(&entry.market_regime)
            .bind(entry.volatility_bucket)
            .bind(entry.drawdown_bucket)
            .bind(entry.momentum_bucket)
            .bind(entry.action_id)
            .bind(entry.q_value)
            .bind(entry.visit_count)
            .bind(entry.total_reward)
            .bind(entry.avg_reward)
            .bind(entry.learning_rate)
            .bind(entry.discount_factor)
            .bind(entry.exploration_rate)
            .execute(&self.pool)
            .await?;
        }
        
        Ok(())
    }
    
    /// Load Q-Learning table from database
    pub async fn load_q_table(&self) -> Result<Vec<QLearningEntry>> {
        let entries = sqlx::query_as::<_, QLearningEntry>(
            "SELECT * FROM q_learning_table ORDER BY q_value DESC"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(entries)
    }
    
    /// Record market regime change
    /// Quinn: "Regime changes are critical events!"
    pub async fn record_regime_change(
        &self,
        regime: &str,
        confidence: Decimal,
        metrics: MarketRegimeMetrics,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO market_regime_history 
             (timestamp, regime, confidence, trend_slope, average_return,
              volatility, volume_surge, correlation_stable, vix_level,
              volume_24h, bid_ask_spread, previous_regime)
             VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
        )
        .bind(regime)
        .bind(confidence)
        .bind(metrics.trend_slope)
        .bind(metrics.average_return)
        .bind(metrics.volatility)
        .bind(metrics.volume_surge)
        .bind(metrics.correlation_stable)
        .bind(metrics.vix_level)
        .bind(metrics.volume_24h)
        .bind(metrics.bid_ask_spread)
        .bind(metrics.previous_regime)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Record performance feedback for learning
    /// Jordan: "Every outcome teaches us something!"
    pub async fn record_performance_feedback(
        &self,
        feedback: PerformanceFeedback,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO performance_feedback 
             (timestamp, market_regime, position_id, signal_id, action_type,
              old_value, new_value, pnl, return_pct, sharpe_contribution,
              max_drawdown, immediate_reward, delayed_reward, total_reward,
              success, success_criteria)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)"
        )
        .bind(feedback.timestamp)
        .bind(&feedback.market_regime)
        .bind(feedback.position_id)
        .bind(&feedback.signal_id)
        .bind(&feedback.action_type)
        .bind(feedback.old_value)
        .bind(feedback.new_value)
        .bind(feedback.pnl)
        .bind(feedback.return_pct)
        .bind(feedback.sharpe_contribution)
        .bind(feedback.max_drawdown)
        .bind(feedback.immediate_reward)
        .bind(feedback.delayed_reward)
        .bind(feedback.total_reward)
        .bind(feedback.success)
        .bind(&feedback.success_criteria)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Get optimal parameters for regime
    /// Casey: "Different markets need different parameters!"
    pub async fn get_regime_parameters(&self, regime: &str) -> Result<HashMap<String, Decimal>> {
        let params = sqlx::query_as::<_, (String, Decimal)>(
            "SELECT parameter_name, current_value 
             FROM adaptive_parameters 
             WHERE optimal_for_regime = $1 OR optimal_for_regime IS NULL
             ORDER BY category, parameter_name"
        )
        .bind(regime)
        .fetch_all(&self.pool)
        .await?;
        
        let mut map = HashMap::new();
        for (name, value) in params {
            map.insert(name, value);
        }
        
        Ok(map)
    }
    
    /// Calculate parameter stability score
    /// Riley: "We need to know which parameters are stable!"
    pub async fn calculate_stability_score(&self, parameter: &str) -> Result<Decimal> {
        let variance: Option<Decimal> = sqlx::query_scalar(
            "SELECT VARIANCE(new_value) 
             FROM auto_tuning_audit 
             WHERE parameter_name = $1 
               AND timestamp > NOW() - INTERVAL '7 days'"
        )
        .bind(parameter)
        .fetch_optional(&self.pool)
        .await?;
        
        // Convert variance to stability score (lower variance = higher stability)
        let stability = match variance {
            Some(v) if v > dec!(0) => dec!(1) / (dec!(1) + v),
            _ => dec!(1),
        };
        
        // Update stability score in database
        sqlx::query(
            "UPDATE adaptive_parameters 
             SET stability_score = $1 
             WHERE parameter_name = $2"
        )
        .bind(stability)
        .bind(parameter)
        .execute(&self.pool)
        .await?;
        
        Ok(stability)
    }
    
    /// Get recent performance metrics
    /// Avery: "Historical performance guides future decisions!"
    pub async fn get_recent_performance(&self, hours: i32) -> Result<PerformanceMetrics> {
        let metrics = sqlx::query_as::<_, PerformanceMetrics>(
            "SELECT 
                AVG(return_pct) as avg_return,
                AVG(sharpe_contribution) as avg_sharpe,
                MAX(max_drawdown) as max_drawdown,
                COUNT(*) FILTER (WHERE success = true)::FLOAT / COUNT(*) as win_rate,
                SUM(pnl) as total_pnl
             FROM performance_feedback
             WHERE timestamp > NOW() - INTERVAL '$1 hours'"
        )
        .bind(hours)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(metrics)
    }
}

/// Market regime detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeMetrics {
    pub trend_slope: Decimal,
    pub average_return: Decimal,
    pub volatility: Decimal,
    pub volume_surge: bool,
    pub correlation_stable: bool,
    pub vix_level: Option<Decimal>,
    pub volume_24h: Decimal,
    pub bid_ask_spread: Decimal,
    pub previous_regime: Option<String>,
}

/// Performance metrics summary
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PerformanceMetrics {
    pub avg_return: Option<Decimal>,
    pub avg_sharpe: Option<Decimal>,
    pub max_drawdown: Option<Decimal>,
    pub win_rate: Option<f64>,
    pub total_pnl: Option<Decimal>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_parameter_persistence() {
        // This would need a test database
        // For now, just verify compilation
        assert!(true);
    }
}