//! Backtesting engine for strategy evaluation

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: u32,
    pub equity_curve: Vec<f64>,
    pub trade_analysis: TradeAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TradeAnalysis {
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub consecutive_wins: u32,
    pub consecutive_losses: u32,
    pub avg_trade_duration: f64,
}

pub struct Backtester;

impl Backtester {
    pub fn new() -> Self {
        Self
    }
    
    pub fn run(&self, strategy: &Value, data: &[Value], initial_capital: f64) -> Result<BacktestResult> {
        if data.is_empty() {
            bail!("No data provided for backtesting");
        }
        
        let strategy_type = strategy["type"].as_str().unwrap_or("momentum");
        let params = &strategy["params"];
        
        match strategy_type {
            "momentum" => self.backtest_momentum(data, initial_capital, params),
            "mean_reversion" => self.backtest_mean_reversion(data, initial_capital, params),
            "trend_following" => self.backtest_trend_following(data, initial_capital, params),
            "market_making" => self.backtest_market_making(data, initial_capital, params),
            _ => self.backtest_simple(data, initial_capital, params),
        }
    }
    
    fn backtest_momentum(&self, data: &[Value], initial_capital: f64, params: &Value) -> Result<BacktestResult> {
        let lookback = params["lookback"].as_u64().unwrap_or(20) as usize;
        let threshold = params["threshold"].as_f64().unwrap_or(0.02);
        
        let mut capital = initial_capital;
        let mut position = 0.0;
        let mut equity_curve = vec![initial_capital];
        let mut trades = Vec::new();
        let mut current_trade = None;
        
        for i in lookback..data.len() {
            let current_price = data[i]["close"].as_f64().unwrap_or(0.0);
            let past_price = data[i - lookback]["close"].as_f64().unwrap_or(1.0);
            let momentum = (current_price / past_price) - 1.0;
            
            // Entry signal
            if position == 0.0 && momentum > threshold {
                position = capital / current_price;
                current_trade = Some(Trade {
                    entry_price: current_price,
                    entry_idx: i,
                    size: position,
                });
            }
            // Exit signal
            else if position > 0.0 && momentum < -threshold / 2.0 {
                if let Some(mut trade) = current_trade.take() {
                    trade.exit_price = Some(current_price);
                    trade.exit_idx = Some(i);
                    trade.pnl = Some((current_price - trade.entry_price) * trade.size);
                    capital += trade.pnl.unwrap();
                    trades.push(trade);
                }
                position = 0.0;
            }
            
            // Update equity
            let current_equity = if position > 0.0 {
                capital + position * (current_price - current_trade.as_ref().unwrap().entry_price)
            } else {
                capital
            };
            equity_curve.push(current_equity);
        }
        
        // Close any open position
        if position > 0.0 && current_trade.is_some() {
            let last_price = data.last().unwrap()["close"].as_f64().unwrap_or(0.0);
            if let Some(mut trade) = current_trade.take() {
                trade.exit_price = Some(last_price);
                trade.exit_idx = Some(data.len() - 1);
                trade.pnl = Some((last_price - trade.entry_price) * trade.size);
                capital += trade.pnl.unwrap();
                trades.push(trade);
            }
        }
        
        self.calculate_metrics(trades, equity_curve, initial_capital)
    }
    
    fn backtest_mean_reversion(&self, data: &[Value], initial_capital: f64, params: &Value) -> Result<BacktestResult> {
        let window = params["window"].as_u64().unwrap_or(20) as usize;
        let z_score_entry = params["z_score_entry"].as_f64().unwrap_or(2.0);
        let z_score_exit = params["z_score_exit"].as_f64().unwrap_or(0.5);
        
        let mut capital = initial_capital;
        let mut position = 0.0;
        let mut equity_curve = vec![initial_capital];
        let mut trades = Vec::new();
        let mut current_trade = None;
        
        for i in window..data.len() {
            let current_price = data[i]["close"].as_f64().unwrap_or(0.0);
            
            // Calculate rolling mean and std
            let window_prices: Vec<f64> = (i - window..i)
                .map(|j| data[j]["close"].as_f64().unwrap_or(0.0))
                .collect();
            
            let mean = window_prices.iter().sum::<f64>() / window as f64;
            let variance = window_prices.iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>() / window as f64;
            let std = variance.sqrt();
            
            let z_score = if std > 0.0 {
                (current_price - mean) / std
            } else {
                0.0
            };
            
            // Entry signal (price too low, expect reversion up)
            if position == 0.0 && z_score < -z_score_entry {
                position = capital / current_price;
                current_trade = Some(Trade {
                    entry_price: current_price,
                    entry_idx: i,
                    size: position,
                });
            }
            // Exit signal (price reverted to mean)
            else if position > 0.0 && z_score.abs() < z_score_exit {
                if let Some(mut trade) = current_trade.take() {
                    trade.exit_price = Some(current_price);
                    trade.exit_idx = Some(i);
                    trade.pnl = Some((current_price - trade.entry_price) * trade.size);
                    capital += trade.pnl.unwrap();
                    trades.push(trade);
                }
                position = 0.0;
            }
            
            // Update equity
            let current_equity = if position > 0.0 {
                capital + position * (current_price - current_trade.as_ref().unwrap().entry_price)
            } else {
                capital
            };
            equity_curve.push(current_equity);
        }
        
        self.calculate_metrics(trades, equity_curve, initial_capital)
    }
    
    fn backtest_trend_following(&self, data: &[Value], initial_capital: f64, params: &Value) -> Result<BacktestResult> {
        let short_ma = params["short_ma"].as_u64().unwrap_or(10) as usize;
        let long_ma = params["long_ma"].as_u64().unwrap_or(30) as usize;
        
        let mut capital = initial_capital;
        let mut position = 0.0;
        let mut equity_curve = vec![initial_capital];
        let mut trades = Vec::new();
        let mut current_trade = None;
        
        for i in long_ma..data.len() {
            let current_price = data[i]["close"].as_f64().unwrap_or(0.0);
            
            // Calculate moving averages
            let short_avg: f64 = (i - short_ma..i)
                .map(|j| data[j]["close"].as_f64().unwrap_or(0.0))
                .sum::<f64>() / short_ma as f64;
            
            let long_avg: f64 = (i - long_ma..i)
                .map(|j| data[j]["close"].as_f64().unwrap_or(0.0))
                .sum::<f64>() / long_ma as f64;
            
            // Entry signal (golden cross)
            if position == 0.0 && short_avg > long_avg {
                position = capital / current_price;
                current_trade = Some(Trade {
                    entry_price: current_price,
                    entry_idx: i,
                    size: position,
                });
            }
            // Exit signal (death cross)
            else if position > 0.0 && short_avg < long_avg {
                if let Some(mut trade) = current_trade.take() {
                    trade.exit_price = Some(current_price);
                    trade.exit_idx = Some(i);
                    trade.pnl = Some((current_price - trade.entry_price) * trade.size);
                    capital += trade.pnl.unwrap();
                    trades.push(trade);
                }
                position = 0.0;
            }
            
            // Update equity
            let current_equity = if position > 0.0 {
                capital + position * (current_price - current_trade.as_ref().unwrap().entry_price)
            } else {
                capital
            };
            equity_curve.push(current_equity);
        }
        
        self.calculate_metrics(trades, equity_curve, initial_capital)
    }
    
    fn backtest_market_making(&self, data: &[Value], initial_capital: f64, params: &Value) -> Result<BacktestResult> {
        let spread = params["spread"].as_f64().unwrap_or(0.002);
        let inventory_limit = params["inventory_limit"].as_f64().unwrap_or(10.0);
        
        let mut capital = initial_capital;
        let mut inventory = 0.0;
        let mut equity_curve = vec![initial_capital];
        let mut trades = Vec::new();
        
        for i in 1..data.len() {
            let mid_price = data[i]["close"].as_f64().unwrap_or(0.0);
            let bid = mid_price * (1.0 - spread / 2.0);
            let ask = mid_price * (1.0 + spread / 2.0);
            
            // Simulate order fills
            let high = data[i]["high"].as_f64().unwrap_or(mid_price);
            let low = data[i]["low"].as_f64().unwrap_or(mid_price);
            
            // Buy if price hits our bid and we have room
            if low <= bid && inventory < inventory_limit {
                let size = (capital * 0.1) / bid; // Use 10% of capital per trade
                inventory += size;
                capital -= size * bid;
                
                trades.push(Trade {
                    entry_price: bid,
                    entry_idx: i,
                    size,
                    exit_price: None,
                    exit_idx: None,
                    pnl: None,
                });
            }
            
            // Sell if price hits our ask and we have inventory
            if high >= ask && inventory > 0.0 {
                let size = inventory.min(inventory_limit / 2.0);
                inventory -= size;
                capital += size * ask;
                
                // Find and close corresponding buy trades
                for trade in trades.iter_mut().rev() {
                    if trade.exit_price.is_none() && trade.size <= size {
                        trade.exit_price = Some(ask);
                        trade.exit_idx = Some(i);
                        trade.pnl = Some((ask - trade.entry_price) * trade.size);
                        break;
                    }
                }
            }
            
            // Update equity (capital + inventory value)
            let current_equity = capital + inventory * mid_price;
            equity_curve.push(current_equity);
        }
        
        self.calculate_metrics(trades, equity_curve, initial_capital)
    }
    
    fn backtest_simple(&self, data: &[Value], initial_capital: f64, params: &Value) -> Result<BacktestResult> {
        // Simple buy and hold strategy as baseline
        let first_price = data[0]["close"].as_f64().unwrap_or(1.0);
        let last_price = data.last().unwrap()["close"].as_f64().unwrap_or(1.0);
        
        let total_return = (last_price / first_price - 1.0) * 100.0;
        let equity_curve: Vec<f64> = data.iter()
            .map(|d| {
                let price = d["close"].as_f64().unwrap_or(1.0);
                initial_capital * (price / first_price)
            })
            .collect();
        
        Ok(BacktestResult {
            total_return,
            sharpe_ratio: total_return / 20.0, // Simplified
            max_drawdown: self.calculate_max_drawdown(&equity_curve),
            win_rate: 100.0,
            profit_factor: total_return.max(0.0),
            total_trades: 1,
            equity_curve,
            trade_analysis: TradeAnalysis {
                winning_trades: 1,
                losing_trades: 0,
                average_win: total_return,
                average_loss: 0.0,
                largest_win: total_return,
                largest_loss: 0.0,
                consecutive_wins: 1,
                consecutive_losses: 0,
                avg_trade_duration: data.len() as f64,
            },
        })
    }
    
    fn calculate_metrics(&self, trades: Vec<Trade>, equity_curve: Vec<f64>, initial_capital: f64) -> Result<BacktestResult> {
        let total_trades = trades.len() as u32;
        
        if total_trades == 0 {
            return Ok(BacktestResult {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                total_trades: 0,
                equity_curve,
                trade_analysis: TradeAnalysis::default(),
            });
        }
        
        let winning_trades: Vec<&Trade> = trades.iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .collect();
        
        let losing_trades: Vec<&Trade> = trades.iter()
            .filter(|t| t.pnl.unwrap_or(0.0) <= 0.0)
            .collect();
        
        let total_profit: f64 = winning_trades.iter()
            .map(|t| t.pnl.unwrap_or(0.0))
            .sum();
        
        let total_loss: f64 = losing_trades.iter()
            .map(|t| t.pnl.unwrap_or(0.0).abs())
            .sum();
        
        let win_rate = (winning_trades.len() as f64 / total_trades as f64) * 100.0;
        let profit_factor = if total_loss > 0.0 { total_profit / total_loss } else { total_profit };
        
        let final_capital = *equity_curve.last().unwrap_or(&initial_capital);
        let total_return = ((final_capital / initial_capital) - 1.0) * 100.0;
        
        // Calculate Sharpe ratio
        let returns = self.calculate_returns(&equity_curve);
        let sharpe_ratio = self.calculate_sharpe(&returns);
        
        // Max drawdown
        let max_drawdown = self.calculate_max_drawdown(&equity_curve);
        
        // Trade analysis
        let trade_analysis = self.analyze_trades(&trades);
        
        Ok(BacktestResult {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            equity_curve,
            trade_analysis,
        })
    }
    
    fn calculate_returns(&self, equity_curve: &[f64]) -> Vec<f64> {
        let mut returns = Vec::new();
        for i in 1..equity_curve.len() {
            returns.push((equity_curve[i] / equity_curve[i - 1]) - 1.0);
        }
        returns
    }
    
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();
        
        if std > 0.0 {
            // Annualized Sharpe ratio (assuming daily returns)
            mean * (252.0_f64).sqrt() / std
        } else {
            0.0
        }
    }
    
    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = equity_curve[0];
        
        for &equity in equity_curve.iter() {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown * 100.0
    }
    
    fn analyze_trades(&self, trades: &[Trade]) -> TradeAnalysis {
        let mut analysis = TradeAnalysis::default();
        
        let mut consecutive_wins = 0;
        let mut consecutive_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;
        
        for trade in trades {
            let pnl = trade.pnl.unwrap_or(0.0);
            
            if pnl > 0.0 {
                analysis.winning_trades += 1;
                analysis.average_win += pnl;
                analysis.largest_win = analysis.largest_win.max(pnl);
                current_wins += 1;
                current_losses = 0;
                consecutive_wins = consecutive_wins.max(current_wins);
            } else {
                analysis.losing_trades += 1;
                analysis.average_loss += pnl.abs();
                analysis.largest_loss = analysis.largest_loss.min(pnl);
                current_losses += 1;
                current_wins = 0;
                consecutive_losses = consecutive_losses.max(current_losses);
            }
            
            if let (Some(exit_idx), entry_idx) = (trade.exit_idx, trade.entry_idx) {
                analysis.avg_trade_duration += (exit_idx - entry_idx) as f64;
            }
        }
        
        if analysis.winning_trades > 0 {
            analysis.average_win /= analysis.winning_trades as f64;
        }
        
        if analysis.losing_trades > 0 {
            analysis.average_loss /= analysis.losing_trades as f64;
        }
        
        if !trades.is_empty() {
            analysis.avg_trade_duration /= trades.len() as f64;
        }
        
        analysis.consecutive_wins = consecutive_wins;
        analysis.consecutive_losses = consecutive_losses;
        
        analysis
    }
}

#[derive(Debug, Clone)]
struct Trade {
    entry_price: f64,
    entry_idx: usize,
    size: f64,
    exit_price: Option<f64>,
    exit_idx: Option<usize>,
    pnl: Option<f64>,
}

impl Default for TradeAnalysis {
    fn default() -> Self {
        Self {
            winning_trades: 0,
            losing_trades: 0,
            average_win: 0.0,
            average_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            avg_trade_duration: 0.0,
        }
    }
}