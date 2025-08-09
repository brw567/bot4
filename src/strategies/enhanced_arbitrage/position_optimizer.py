"""
Advanced position size optimization for arbitrage and statistical arbitrage.
Implements portfolio-level optimization with risk constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import minimize, LinearConstraint
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionConstraints:
    """Constraints for position sizing"""
    min_position_usd: float = 100
    max_position_usd: float = 10000
    max_position_pct: float = 0.2  # Max 20% per position
    max_gross_exposure: float = 2.0  # Max 200% gross
    max_net_exposure: float = 1.0  # Max 100% net
    max_var_95: float = 0.05  # Max 5% daily VaR
    max_correlation_exposure: float = 0.6  # Max 60% in correlated assets
    min_sharpe_ratio: float = 1.5  # Minimum required Sharpe


@dataclass
class OpportunityInput:
    """Input data for an opportunity"""
    id: str
    expected_return: float  # Expected return percentage
    volatility: float  # Return volatility
    sharpe_ratio: float
    min_size_usd: float
    max_size_usd: float
    confidence: float  # Confidence in the opportunity (0-1)
    correlation_group: Optional[str] = None  # For grouping correlated opportunities


class PositionOptimizer:
    """
    Optimizes position sizes across multiple opportunities.
    Uses modern portfolio theory with practical constraints.
    """
    
    def __init__(self, 
                 total_capital: float,
                 constraints: Optional[PositionConstraints] = None):
        """
        Initialize the optimizer.
        
        Args:
            total_capital: Total available capital
            constraints: Position constraints
        """
        self.total_capital = total_capital
        self.constraints = constraints or PositionConstraints()
        
        # Risk-free rate (annualized)
        self.risk_free_rate = 0.03
        
        # Optimization parameters
        self.solver_tolerance = 1e-6
        self.max_iterations = 1000
    
    def optimize_positions(self, 
                         opportunities: List[OpportunityInput],
                         correlation_matrix: Optional[np.ndarray] = None,
                         method: str = 'mean_variance') -> Dict[str, float]:
        """
        Optimize position sizes for given opportunities.
        
        Args:
            opportunities: List of opportunities to size
            correlation_matrix: Correlation between opportunities
            method: Optimization method ('mean_variance', 'risk_parity', 'kelly')
            
        Returns:
            Dictionary of opportunity_id -> position_size_usd
        """
        if len(opportunities) == 0:
            return {}
        
        if len(opportunities) == 1:
            # Single opportunity, use simple sizing
            return self._size_single_opportunity(opportunities[0])
        
        # Use specified method
        if method == 'mean_variance':
            return self._optimize_mean_variance(opportunities, correlation_matrix)
        elif method == 'risk_parity':
            return self._optimize_risk_parity(opportunities, correlation_matrix)
        elif method == 'kelly':
            return self._optimize_kelly(opportunities, correlation_matrix)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _size_single_opportunity(self, opp: OpportunityInput) -> Dict[str, float]:
        """Size a single opportunity"""
        # Base size on Kelly criterion
        kelly_fraction = self._calculate_kelly_fraction(
            opp.expected_return, 
            opp.volatility,
            opp.confidence
        )
        
        # Apply constraints
        position_size = kelly_fraction * self.total_capital
        position_size = max(opp.min_size_usd, position_size)
        position_size = min(opp.max_size_usd, position_size)
        position_size = min(
            position_size,
            self.total_capital * self.constraints.max_position_pct
        )
        
        # Check minimum size
        if position_size < self.constraints.min_position_usd:
            return {}
        
        return {opp.id: position_size}
    
    def _optimize_mean_variance(self, 
                              opportunities: List[OpportunityInput],
                              correlation_matrix: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Mean-variance optimization using cvxpy.
        Maximizes Sharpe ratio subject to constraints.
        """
        n = len(opportunities)
        
        # Extract data
        returns = np.array([opp.expected_return for opp in opportunities])
        volatilities = np.array([opp.volatility for opp in opportunities])
        
        # Build covariance matrix
        if correlation_matrix is None:
            # Assume low correlation
            correlation_matrix = np.eye(n) * 0.9 + np.ones((n, n)) * 0.1
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Decision variables (weights)
        weights = cp.Variable(n)
        
        # Expected portfolio return
        portfolio_return = returns @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Objective: Maximize Sharpe ratio (approximated)
        # We use return - 0.5 * lambda * variance (risk-adjusted return)
        risk_aversion = 2  # Risk aversion parameter
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = []
        
        # Budget constraint
        constraints.append(cp.sum(weights) <= 1.0)
        
        # Non-negative weights (no shorting)
        constraints.append(weights >= 0)
        
        # Position limits
        max_weight = self.constraints.max_position_pct
        constraints.append(weights <= max_weight)
        
        # Minimum position size
        # Note: We enforce minimum size after optimization instead of as constraint
        # CVXPY doesn't support logical OR constraints directly
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                # Convert weights to positions
                positions = {}
                for i, opp in enumerate(opportunities):
                    weight = weights.value[i]
                    if weight > min_weight:
                        position_size = weight * self.total_capital
                        
                        # Apply opportunity-specific constraints
                        position_size = max(opp.min_size_usd, position_size)
                        position_size = min(opp.max_size_usd, position_size)
                        
                        positions[opp.id] = position_size
                
                return positions
            else:
                logger.warning(f"Optimization failed: {problem.status}")
                return self._fallback_sizing(opportunities)
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._fallback_sizing(opportunities)
    
    def _optimize_risk_parity(self,
                            opportunities: List[OpportunityInput],
                            correlation_matrix: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Risk parity optimization - equal risk contribution from each position.
        """
        n = len(opportunities)
        volatilities = np.array([opp.volatility for opp in opportunities])
        
        # Build covariance matrix
        if correlation_matrix is None:
            correlation_matrix = np.eye(n) * 0.9 + np.ones((n, n)) * 0.1
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Initial guess: inverse volatility weighting
        x0 = 1 / volatilities
        x0 = x0 / x0.sum()
        
        # Objective: minimize variance subject to equal risk contribution
        def objective(weights):
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib
            
            # Deviation from equal contribution
            target_contrib = portfolio_var / n
            deviation = np.sum((risk_contrib - target_contrib) ** 2)
            
            return deviation
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative
        ]
        
        # Bounds
        bounds = [(0, self.constraints.max_position_pct) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': self.solver_tolerance}
        )
        
        if result.success:
            positions = {}
            for i, opp in enumerate(opportunities):
                weight = result.x[i]
                if weight > 0.001:  # Minimum threshold
                    position_size = weight * self.total_capital
                    
                    # Apply constraints
                    position_size = max(opp.min_size_usd, position_size)
                    position_size = min(opp.max_size_usd, position_size)
                    
                    if position_size >= self.constraints.min_position_usd:
                        positions[opp.id] = position_size
            
            return positions
        else:
            return self._fallback_sizing(opportunities)
    
    def _optimize_kelly(self,
                       opportunities: List[OpportunityInput],
                       correlation_matrix: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Multi-asset Kelly criterion optimization.
        """
        positions = {}
        
        # For correlated assets, reduce Kelly fraction
        if correlation_matrix is not None:
            avg_correlation = (correlation_matrix.sum() - len(opportunities)) / (
                len(opportunities) * (len(opportunities) - 1)
            )
            correlation_penalty = 1 - avg_correlation * 0.5
        else:
            correlation_penalty = 1.0
        
        # Calculate Kelly size for each opportunity
        for opp in opportunities:
            kelly_fraction = self._calculate_kelly_fraction(
                opp.expected_return,
                opp.volatility,
                opp.confidence
            )
            
            # Apply correlation penalty
            kelly_fraction *= correlation_penalty
            
            # Apply fractional Kelly (25% default)
            kelly_fraction *= 0.25
            
            # Calculate position size
            position_size = kelly_fraction * self.total_capital
            
            # Apply constraints
            position_size = max(opp.min_size_usd, position_size)
            position_size = min(opp.max_size_usd, position_size)
            position_size = min(
                position_size,
                self.total_capital * self.constraints.max_position_pct
            )
            
            if position_size >= self.constraints.min_position_usd:
                positions[opp.id] = position_size
        
        # Check total exposure
        total_exposure = sum(positions.values())
        if total_exposure > self.total_capital * self.constraints.max_gross_exposure:
            # Scale down proportionally
            scale = (self.total_capital * self.constraints.max_gross_exposure) / total_exposure
            positions = {k: v * scale for k, v in positions.items()}
        
        return positions
    
    def _calculate_kelly_fraction(self, expected_return: float,
                                volatility: float, confidence: float) -> float:
        """Calculate Kelly fraction for a single opportunity"""
        if volatility <= 0:
            return 0
        
        # Kelly formula: f = μ / σ²
        # Where μ is expected excess return and σ is volatility
        excess_return = expected_return - self.risk_free_rate / 252  # Daily
        kelly = excess_return / (volatility ** 2)
        
        # Apply confidence adjustment
        kelly *= confidence
        
        # Cap at 1 (no leverage)
        return min(max(0, kelly), 1.0)
    
    def _fallback_sizing(self, opportunities: List[OpportunityInput]) -> Dict[str, float]:
        """Simple fallback sizing when optimization fails"""
        positions = {}
        
        # Sort by Sharpe ratio
        sorted_opps = sorted(
            opportunities, 
            key=lambda x: x.sharpe_ratio * x.confidence,
            reverse=True
        )
        
        # Allocate capital to best opportunities
        remaining_capital = self.total_capital * 0.8  # Keep 20% reserve
        
        for opp in sorted_opps:
            if remaining_capital < self.constraints.min_position_usd:
                break
            
            # Size based on confidence and constraints
            position_size = min(
                remaining_capital * 0.3,  # Max 30% of remaining
                opp.max_size_usd,
                self.total_capital * self.constraints.max_position_pct
            )
            
            position_size = max(position_size, opp.min_size_usd)
            
            if position_size >= self.constraints.min_position_usd:
                positions[opp.id] = position_size
                remaining_capital -= position_size
        
        return positions
    
    def calculate_portfolio_metrics(self,
                                  positions: Dict[str, float],
                                  opportunities: List[OpportunityInput],
                                  correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate portfolio-level metrics for given positions"""
        if not positions:
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'var_95': 0,
                'total_exposure': 0
            }
        
        # Map opportunities by ID
        opp_map = {opp.id: opp for opp in opportunities}
        
        # Calculate weights
        total_position = sum(positions.values())
        weights = np.array([
            positions.get(opp.id, 0) / self.total_capital
            for opp in opportunities
        ])
        
        # Expected return
        returns = np.array([opp.expected_return for opp in opportunities])
        portfolio_return = np.dot(weights, returns)
        
        # Volatility
        volatilities = np.array([opp.volatility for opp in opportunities])
        
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(opportunities))
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate / 252
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # VaR (95%)
        var_95 = portfolio_volatility * 1.645  # Normal distribution
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
            'var_95': var_95,
            'total_exposure': total_position,
            'exposure_pct': total_position / self.total_capital
        }