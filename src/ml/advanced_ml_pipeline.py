"""
Advanced ML Pipeline for RC1 Trading Bot
Implements feature engineering, model training, and real-time inference
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
import optuna
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
# Torch imports removed - using CPU-optimized models only
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging



@dataclass
class MarketFeatures:
    """Market feature container"""
    timestamp: datetime
    ohlcv: Dict[str, float]
    technical_indicators: Dict[str, float]
    market_microstructure: Dict[str, float]
    sentiment_scores: Dict[str, float]
    on_chain_metrics: Optional[Dict[str, float]] = None

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for trading signals"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = RobustScaler()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Market microstructure
        df = self._add_microstructure_features(df)
        
        # Time-based features
        df = self._add_temporal_features(df)
        
        # Rolling statistics
        df = self._add_rolling_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Clean up
        df = df.dropna()
        
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'target']]
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features"""
        # Returns
        for period in [1, 5, 15, 30, 60]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume-weighted average price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['close_vwap_ratio'] = df['close'] / df['vwap']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        # Trend indicators
        for period in [10, 20, 50, 200]:
            df[f'sma_{period}'] = SMAIndicator(df['close'], period).sma_indicator()
            df[f'ema_{period}'] = EMAIndicator(df['close'], period).ema_indicator()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = RSIIndicator(df['close'], period).rsi()
        
        # Bollinger Bands
        for period in [20, 50]:
            bb = BollingerBands(df['close'], period)
            df[f'bb_high_{period}'] = bb.bollinger_hband()
            df[f'bb_low_{period}'] = bb.bollinger_lband()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_low_{period}']) / df[f'bb_width_{period}']
        
        # ATR
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread (if available)
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = df['spread'] / df['close']
            df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Order flow imbalance (if order book data available)
        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            df['order_flow_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'])
        
        # Price impact
        df['price_impact'] = df['close'].diff() / df['volume'].rolling(10).mean()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Price statistics
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            
            # Volume statistics
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # High-low range
            df[f'range_mean_{window}'] = (df['high'] - df['low']).rolling(window).mean()
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions"""
        # Price momentum * volume
        df['price_volume_interaction'] = df['return_5'] * df['volume_ratio']
        
        # RSI divergence
        if 'rsi_14' in df.columns:
            df['rsi_price_divergence'] = df['rsi_14'] - (df['close'].pct_change(14) * 100 + 50)
        
        # Volatility regime
        df['volatility_regime'] = pd.qcut(df['atr'], q=3, labels=['low', 'medium', 'high'])
        df = pd.get_dummies(df, columns=['volatility_regime'], prefix='vol_regime')
        
        return df


class MLModelEnsemble:
    """Ensemble of ML models for trading predictions"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
    
    def _create_models(self) -> Dict[str, Any]:
        """Create ensemble of models"""
        return {
            'lgb_classifier': lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            ),
            'lgb_regressor': lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            ),
            'rf_classifier': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb_regressor': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=5,
                random_state=42
            )
        }
    
    def train(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'target',
        optimize_hyperparams: bool = False
    ) -> Dict[str, float]:
        """Train ensemble models"""
        
        # Feature engineering
        df_features = self.feature_engineer.create_features(df.copy())
        
        # Prepare data
        X = df_features[self.feature_engineer.feature_columns]
        y = df_features[target_col]
        
        # Scale features
        X_scaled = self.feature_engineer.scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize models
        self.models = self._create_models()
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            self._optimize_hyperparameters(X_scaled, y, tscv)
        
        # Train models
        scores = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy' if 'classifier' in name else 'neg_mean_squared_error')
            scores[name] = np.mean(cv_scores)
            
            # Final training
            model.fit(X_scaled, y)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': self.feature_engineer.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"\nTop 10 features for {name}:")
                logger.info(importances.head(10))
        
        self.is_fitted = True
        return scores
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, cv):
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            }
            
            model = lgb.LGBMClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Update model with best params
        best_params = study.best_params
        self.models['lgb_classifier'] = lgb.LGBMClassifier(**best_params, random_state=42)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from ensemble"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")
        
        # Feature engineering
        df_features = self.feature_engineer.create_features(df.copy())
        X = df_features[self.feature_engineer.feature_columns]
        X_scaled = self.feature_engineer.scaler.transform(X)
        
        # Generate predictions
        predictions = {}
        for name, model in self.models.items():
            if 'classifier' in name:
                # Get probability of positive class
                predictions[name] = model.predict_proba(X_scaled)[:, 1]
            else:
                predictions[name] = model.predict(X_scaled)
        
        # Ensemble prediction (weighted average)
        weights = {'lgb_classifier': 0.3, 'lgb_regressor': 0.3, 'rf_classifier': 0.2, 'gb_regressor': 0.2}
        ensemble_pred = np.zeros(len(X))
        
        for name, pred in predictions.items():
            ensemble_pred += pred * weights.get(name, 0.25)
        
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def save_models(self, path: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_engineer.feature_columns,
            'scaler': self.feature_engineer.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, f"{path}/ml_ensemble.pkl")
    
    def load_models(self, path: str):
        """Load trained models"""
        model_data = joblib.load(f"{path}/ml_ensemble.pkl")
        self.models = model_data['models']
        self.feature_engineer.feature_columns = model_data['feature_columns']
        self.feature_engineer.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']


class CPUOptimizedPredictor:
    """CPU-optimized predictor using XGBoost/LightGBM instead of deep learning"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            max_depth=-1,
            min_child_weight=0.001,
            min_split_gain=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_jobs=-1,
            random_state=42
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)


class RealTimeInference:
    """Real-time ML inference engine"""
    
    def __init__(self, model_path: str):
        self.ensemble = MLModelEnsemble()
        try:
            self.ensemble.load_models(model_path)
        except FileNotFoundError:
            # Models not trained yet - will train on first data
            pass
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_cache = {}
        self.cache_ttl = 60  # seconds
    
    async def predict_async(
        self, 
        market_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, float]:
        """Async prediction with caching"""
        
        # Check cache
        cache_key = self._generate_cache_key(market_data)
        if use_cache and cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_result
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            self.executor,
            self._predict_sync,
            market_data
        )
        
        # Cache result
        if use_cache:
            self.prediction_cache[cache_key] = (predictions, datetime.now())
        
        return predictions
    
    def _predict_sync(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Synchronous prediction"""
        # Convert market data to DataFrame
        df = pd.DataFrame([market_data])
        
        # Get predictions
        predictions = self.ensemble.predict(df)
        
        # Convert to trading signals
        signals = {}
        for name, pred in predictions.items():
            signal_strength = float(pred[0])
            signals[name] = {
                'signal': 'buy' if signal_strength > 0.6 else 'sell' if signal_strength < 0.4 else 'hold',
                'confidence': abs(signal_strength - 0.5) * 2,
                'raw_score': signal_strength
            }
        
        return signals
    
    def _generate_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key from market data"""
        return f"{market_data.get('pair')}_{market_data.get('timestamp')}"


# Usage example
async def example_ml_pipeline():
    # Load historical data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.rand(1000) * 1000000,
        'target': (np.random.rand(1000) > 0.5).astype(int)
    })
    
    # Train ensemble
    ensemble = MLModelEnsemble()
    scores = ensemble.train(df, optimize_hyperparams=False)
    logger.info(f"Model scores: {scores}")
    
    # Save models
    ensemble.save_models("./models")
    
    # Real-time inference
    inference = RealTimeInference("./models")
    
    # Simulate real-time prediction
    market_data = {
        'timestamp': datetime.now(),
        'open': 50000,
        'high': 50200,
        'low': 49800,
        'close': 50100,
        'volume': 1000000,
        'pair': 'BTC/USDT'
    }
    
    signals = await inference.predict_async(market_data)
    logger.info(f"Trading signals: {signals}")


if __name__ == "__main__":
    asyncio.run(example_ml_pipeline())