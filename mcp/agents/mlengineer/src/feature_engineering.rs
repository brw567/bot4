//! Feature engineering for ML models

use anyhow::{Result, bail};
use ndarray::{Array2, Axis};
use serde_json::Value;
use ta::{indicators::*, DataItem};

pub struct FeatureEngineer;

impl FeatureEngineer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn extract(&self, candles: &[Value], feature_set: &str) -> Result<Array2<f64>> {
        if candles.is_empty() {
            bail!("No candles provided for feature extraction");
        }
        
        let mut features = Vec::new();
        
        // Convert to price series
        let prices: Vec<f64> = candles.iter()
            .filter_map(|c| c["close"].as_f64())
            .collect();
        
        let highs: Vec<f64> = candles.iter()
            .filter_map(|c| c["high"].as_f64())
            .collect();
        
        let lows: Vec<f64> = candles.iter()
            .filter_map(|c| c["low"].as_f64())
            .collect();
        
        let volumes: Vec<f64> = candles.iter()
            .filter_map(|c| c["volume"].as_f64())
            .collect();
        
        match feature_set {
            "technical" => {
                features.extend(self.extract_technical_features(&prices, &highs, &lows, &volumes)?);
            }
            "statistical" => {
                features.extend(self.extract_statistical_features(&prices)?);
            }
            "all" | "default" => {
                features.extend(self.extract_technical_features(&prices, &highs, &lows, &volumes)?);
                features.extend(self.extract_statistical_features(&prices)?);
                features.extend(self.extract_microstructure_features(&candles)?);
            }
            _ => {
                features.extend(self.extract_basic_features(&prices)?);
            }
        }
        
        // Convert to ndarray
        let n_samples = candles.len();
        let n_features = features.len() / n_samples;
        
        Array2::from_shape_vec((n_samples, n_features), features)
            .map_err(|e| anyhow::anyhow!("Failed to create feature array: {}", e))
    }
    
    pub fn get_feature_names(&self, feature_set: &str) -> Vec<String> {
        match feature_set {
            "technical" => vec![
                "sma_20".to_string(),
                "ema_20".to_string(),
                "rsi_14".to_string(),
                "macd".to_string(),
                "macd_signal".to_string(),
                "bb_upper".to_string(),
                "bb_lower".to_string(),
                "atr_14".to_string(),
                "obv".to_string(),
                "stoch_k".to_string(),
                "stoch_d".to_string(),
            ],
            "statistical" => vec![
                "return_1".to_string(),
                "return_5".to_string(),
                "volatility_20".to_string(),
                "skewness".to_string(),
                "kurtosis".to_string(),
                "hurst_exponent".to_string(),
            ],
            "all" | "default" => {
                let mut names = self.get_feature_names("technical");
                names.extend(self.get_feature_names("statistical"));
                names.extend(vec![
                    "bid_ask_spread".to_string(),
                    "order_imbalance".to_string(),
                    "volume_profile".to_string(),
                ]);
                names
            }
            _ => vec![
                "price".to_string(),
                "volume".to_string(),
                "return".to_string(),
            ]
        }
    }
    
    fn extract_technical_features(&self, prices: &[f64], highs: &[f64], 
                                  lows: &[f64], volumes: &[f64]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        let n = prices.len();
        
        // SMA 20
        let sma = self.calculate_sma(prices, 20);
        features.extend(sma);
        
        // EMA 20
        let ema = self.calculate_ema(prices, 20);
        features.extend(ema);
        
        // RSI 14
        let rsi = self.calculate_rsi(prices, 14);
        features.extend(rsi);
        
        // MACD
        let (macd, signal) = self.calculate_macd(prices);
        features.extend(macd);
        features.extend(signal);
        
        // Bollinger Bands
        let (bb_upper, bb_lower) = self.calculate_bollinger_bands(prices, 20, 2.0);
        features.extend(bb_upper);
        features.extend(bb_lower);
        
        // ATR
        let atr = self.calculate_atr(highs, lows, prices, 14);
        features.extend(atr);
        
        // OBV
        let obv = self.calculate_obv(prices, volumes);
        features.extend(obv);
        
        // Stochastic
        let (stoch_k, stoch_d) = self.calculate_stochastic(highs, lows, prices, 14);
        features.extend(stoch_k);
        features.extend(stoch_d);
        
        Ok(features)
    }
    
    fn extract_statistical_features(&self, prices: &[f64]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Returns
        let returns_1 = self.calculate_returns(prices, 1);
        features.extend(returns_1);
        
        let returns_5 = self.calculate_returns(prices, 5);
        features.extend(returns_5);
        
        // Rolling volatility
        let volatility = self.calculate_rolling_volatility(prices, 20);
        features.extend(volatility);
        
        // Skewness
        let skewness = self.calculate_rolling_skewness(prices, 20);
        features.extend(skewness);
        
        // Kurtosis
        let kurtosis = self.calculate_rolling_kurtosis(prices, 20);
        features.extend(kurtosis);
        
        // Hurst exponent
        let hurst = self.calculate_hurst_exponent(prices, 20);
        features.extend(hurst);
        
        Ok(features)
    }
    
    fn extract_microstructure_features(&self, candles: &[Value]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        for candle in candles {
            // Bid-ask spread proxy (high - low)
            let high = candle["high"].as_f64().unwrap_or(0.0);
            let low = candle["low"].as_f64().unwrap_or(0.0);
            let spread = (high - low) / ((high + low) / 2.0);
            features.push(spread);
            
            // Order imbalance proxy (close - open)
            let close = candle["close"].as_f64().unwrap_or(0.0);
            let open = candle["open"].as_f64().unwrap_or(0.0);
            let imbalance = (close - open) / open;
            features.push(imbalance);
            
            // Volume profile
            let volume = candle["volume"].as_f64().unwrap_or(0.0);
            let typical_price = (high + low + close) / 3.0;
            let volume_weighted = volume * typical_price;
            features.push(volume_weighted);
        }
        
        Ok(features)
    }
    
    fn extract_basic_features(&self, prices: &[f64]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        for (i, &price) in prices.iter().enumerate() {
            features.push(price);
            
            // Simple return
            if i > 0 {
                features.push((price / prices[i - 1]) - 1.0);
            } else {
                features.push(0.0);
            }
            
            // Normalized price
            let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let norm_price = (price - min_price) / (max_price - min_price + 1e-10);
            features.push(norm_price);
        }
        
        Ok(features)
    }
    
    // Technical indicator calculations
    fn calculate_sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut sma = vec![0.0; prices.len()];
        
        for i in period..prices.len() {
            let sum: f64 = prices[i - period..i].iter().sum();
            sma[i] = sum / period as f64;
        }
        
        // Fill initial values
        for i in 0..period.min(prices.len()) {
            sma[i] = prices[..i + 1].iter().sum::<f64>() / (i + 1) as f64;
        }
        
        sma
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut ema = vec![0.0; prices.len()];
        if prices.is_empty() {
            return ema;
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        ema[0] = prices[0];
        
        for i in 1..prices.len() {
            ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
        }
        
        ema
    }
    
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut rsi = vec![50.0; prices.len()];
        if prices.len() < period + 1 {
            return rsi;
        }
        
        let mut gains = vec![0.0; prices.len()];
        let mut losses = vec![0.0; prices.len()];
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }
        
        let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;
        
        for i in period..prices.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            
            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                rsi[i] = 100.0;
            }
        }
        
        rsi
    }
    
    fn calculate_macd(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ema_12 = self.calculate_ema(prices, 12);
        let ema_26 = self.calculate_ema(prices, 26);
        
        let mut macd = vec![0.0; prices.len()];
        for i in 0..prices.len() {
            macd[i] = ema_12[i] - ema_26[i];
        }
        
        let signal = self.calculate_ema(&macd, 9);
        
        (macd, signal)
    }
    
    fn calculate_bollinger_bands(&self, prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>) {
        let sma = self.calculate_sma(prices, period);
        let mut upper = vec![0.0; prices.len()];
        let mut lower = vec![0.0; prices.len()];
        
        for i in period..prices.len() {
            let variance: f64 = prices[i - period..i].iter()
                .map(|p| (p - sma[i]).powi(2))
                .sum::<f64>() / period as f64;
            let std = variance.sqrt();
            
            upper[i] = sma[i] + std_dev * std;
            lower[i] = sma[i] - std_dev * std;
        }
        
        // Fill initial values
        for i in 0..period.min(prices.len()) {
            upper[i] = prices[i] * 1.02;
            lower[i] = prices[i] * 0.98;
        }
        
        (upper, lower)
    }
    
    fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        let mut tr = vec![0.0; closes.len()];
        let mut atr = vec![0.0; closes.len()];
        
        if closes.is_empty() {
            return atr;
        }
        
        // First TR is just high - low
        tr[0] = highs[0] - lows[0];
        
        for i in 1..closes.len() {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
        
        // Calculate ATR as EMA of TR
        atr[0] = tr[0];
        for i in 1..closes.len() {
            if i < period {
                atr[i] = tr[..=i].iter().sum::<f64>() / (i + 1) as f64;
            } else {
                atr[i] = (atr[i - 1] * (period - 1) as f64 + tr[i]) / period as f64;
            }
        }
        
        atr
    }
    
    fn calculate_obv(&self, prices: &[f64], volumes: &[f64]) -> Vec<f64> {
        let mut obv = vec![0.0; prices.len()];
        
        if prices.is_empty() {
            return obv;
        }
        
        obv[0] = volumes[0];
        
        for i in 1..prices.len() {
            if prices[i] > prices[i - 1] {
                obv[i] = obv[i - 1] + volumes[i];
            } else if prices[i] < prices[i - 1] {
                obv[i] = obv[i - 1] - volumes[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }
        
        obv
    }
    
    fn calculate_stochastic(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
        let mut k = vec![50.0; closes.len()];
        
        for i in period..closes.len() {
            let high_period = highs[i - period..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let low_period = lows[i - period..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            
            if high_period > low_period {
                k[i] = 100.0 * (closes[i] - low_period) / (high_period - low_period);
            }
        }
        
        let d = self.calculate_sma(&k, 3);
        
        (k, d)
    }
    
    // Statistical calculations
    fn calculate_returns(&self, prices: &[f64], lag: usize) -> Vec<f64> {
        let mut returns = vec![0.0; prices.len()];
        
        for i in lag..prices.len() {
            returns[i] = (prices[i] / prices[i - lag]) - 1.0;
        }
        
        returns
    }
    
    fn calculate_rolling_volatility(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let returns = self.calculate_returns(prices, 1);
        let mut volatility = vec![0.0; prices.len()];
        
        for i in window..prices.len() {
            let window_returns = &returns[i - window..i];
            let mean = window_returns.iter().sum::<f64>() / window as f64;
            let variance = window_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / window as f64;
            volatility[i] = variance.sqrt();
        }
        
        volatility
    }
    
    fn calculate_rolling_skewness(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let returns = self.calculate_returns(prices, 1);
        let mut skewness = vec![0.0; prices.len()];
        
        for i in window..prices.len() {
            let window_returns = &returns[i - window..i];
            let mean = window_returns.iter().sum::<f64>() / window as f64;
            let variance = window_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / window as f64;
            let std = variance.sqrt();
            
            if std > 0.0 {
                let skew = window_returns.iter()
                    .map(|r| ((r - mean) / std).powi(3))
                    .sum::<f64>() / window as f64;
                skewness[i] = skew;
            }
        }
        
        skewness
    }
    
    fn calculate_rolling_kurtosis(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let returns = self.calculate_returns(prices, 1);
        let mut kurtosis = vec![0.0; prices.len()];
        
        for i in window..prices.len() {
            let window_returns = &returns[i - window..i];
            let mean = window_returns.iter().sum::<f64>() / window as f64;
            let variance = window_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / window as f64;
            let std = variance.sqrt();
            
            if std > 0.0 {
                let kurt = window_returns.iter()
                    .map(|r| ((r - mean) / std).powi(4))
                    .sum::<f64>() / window as f64 - 3.0;
                kurtosis[i] = kurt;
            }
        }
        
        kurtosis
    }
    
    fn calculate_hurst_exponent(&self, prices: &[f64], window: usize) -> Vec<f64> {
        // Simplified Hurst exponent calculation using R/S analysis
        let mut hurst = vec![0.5; prices.len()];
        
        for i in window..prices.len() {
            let window_prices = &prices[i - window..=i];
            let returns: Vec<f64> = (1..window_prices.len())
                .map(|j| (window_prices[j] / window_prices[j - 1]).ln())
                .collect();
            
            if returns.is_empty() {
                continue;
            }
            
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let mut cumsum = vec![0.0; returns.len()];
            cumsum[0] = returns[0] - mean;
            
            for j in 1..returns.len() {
                cumsum[j] = cumsum[j - 1] + returns[j] - mean;
            }
            
            let range = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max) -
                       cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
            
            let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();
            
            if std > 0.0 && range > 0.0 {
                // Simplified Hurst estimate
                hurst[i] = 0.5 + 0.5 * (range / std).ln() / (returns.len() as f64).ln();
                hurst[i] = hurst[i].max(0.0).min(1.0);
            }
        }
        
        hurst
    }
}