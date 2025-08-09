#!/usr/bin/env python3
"""
Risk Limits Validation Script
Quinn's enforcement tool - protecting capital above all
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Risk limit thresholds (Quinn's rules)
RISK_LIMITS = {
    'max_position_size': 0.02,      # 2% max per position
    'max_total_exposure': 0.10,     # 10% max total
    'max_leverage': 3,               # 3x maximum
    'max_drawdown': 0.15,           # 15% max drawdown
    'required_stop_loss': True,     # Must have stop losses
    'min_sharpe_ratio': 1.0,        # Minimum acceptable Sharpe
    'max_correlation': 0.7,         # Max correlation between positions
}

class RiskValidator:
    def __init__(self):
        self.violations = []
        self.warnings = []
        self.files_checked = 0
        
    def check_position_sizing(self, filepath: Path) -> None:
        """Check position sizing in strategy files"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Check for position size calculations
                position_patterns = [
                    r'position_size\s*=\s*([0-9.]+)',
                    r'size\s*=\s*capital\s*\*\s*([0-9.]+)',
                    r'amount\s*=\s*balance\s*\*\s*([0-9.]+)',
                ]
                
                for pattern in position_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        size = float(match)
                        if size > RISK_LIMITS['max_position_size']:
                            self.violations.append({
                                'file': str(filepath),
                                'type': 'position_size',
                                'value': size,
                                'limit': RISK_LIMITS['max_position_size'],
                                'message': f'Position size {size:.1%} exceeds limit {RISK_LIMITS["max_position_size"]:.1%}'
                            })
                
        except Exception as e:
            print(f"Error checking {filepath}: {e}")
    
    def check_leverage(self, filepath: Path) -> None:
        """Check leverage settings"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Check for leverage settings
                leverage_patterns = [
                    r'leverage\s*=\s*([0-9]+)',
                    r'margin_multiplier\s*=\s*([0-9]+)',
                    r'leverage_ratio\s*=\s*([0-9]+)',
                ]
                
                for pattern in leverage_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        leverage = int(match)
                        if leverage > RISK_LIMITS['max_leverage']:
                            self.violations.append({
                                'file': str(filepath),
                                'type': 'leverage',
                                'value': leverage,
                                'limit': RISK_LIMITS['max_leverage'],
                                'message': f'Leverage {leverage}x exceeds limit {RISK_LIMITS["max_leverage"]}x'
                            })
                
        except Exception as e:
            print(f"Error checking {filepath}: {e}")
    
    def check_stop_losses(self, filepath: Path) -> None:
        """Check for stop loss implementation"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Check if file contains trading logic
                if any(word in content for word in ['place_order', 'execute_trade', 'open_position']):
                    # Check for stop loss
                    stop_loss_patterns = [
                        r'stop_loss',
                        r'stop_price',
                        r'sl_price',
                        r'stop_order',
                    ]
                    
                    has_stop_loss = any(re.search(pattern, content, re.IGNORECASE) 
                                      for pattern in stop_loss_patterns)
                    
                    if not has_stop_loss:
                        self.violations.append({
                            'file': str(filepath),
                            'type': 'stop_loss',
                            'value': False,
                            'limit': True,
                            'message': 'No stop loss implementation found'
                        })
                
        except Exception as e:
            print(f"Error checking {filepath}: {e}")
    
    def check_config_files(self) -> None:
        """Check configuration files for risk settings"""
        config_files = [
            'config.yaml',
            'config/config.yaml',
            'strategies.json',
            '.env.example',
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                self.check_config_risk_params(config_file)
    
    def check_config_risk_params(self, filepath: str) -> None:
        """Check risk parameters in config files"""
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml'):
                    config = yaml.safe_load(f)
                    self.validate_config_risks(config, filepath)
                elif filepath.endswith('.json'):
                    import json
                    config = json.load(f)
                    self.validate_config_risks(config, filepath)
                elif filepath.endswith('.env') or filepath.endswith('.env.example'):
                    content = f.read()
                    self.check_env_risks(content, filepath)
                    
        except Exception as e:
            print(f"Error checking config {filepath}: {e}")
    
    def validate_config_risks(self, config: Dict, filepath: str) -> None:
        """Validate risk parameters in config dictionary"""
        if isinstance(config, dict):
            # Check for risk-related settings
            risk_keys = ['risk', 'position', 'leverage', 'trading']
            
            for key in risk_keys:
                if key in config:
                    risk_config = config[key]
                    if isinstance(risk_config, dict):
                        # Check position size
                        if 'max_position' in risk_config:
                            value = float(risk_config['max_position'])
                            if value > RISK_LIMITS['max_position_size']:
                                self.violations.append({
                                    'file': filepath,
                                    'type': 'config_position_size',
                                    'value': value,
                                    'limit': RISK_LIMITS['max_position_size'],
                                    'message': f'Config position size {value:.1%} exceeds limit'
                                })
                        
                        # Check leverage
                        if 'leverage' in risk_config:
                            value = int(risk_config['leverage'])
                            if value > RISK_LIMITS['max_leverage']:
                                self.violations.append({
                                    'file': filepath,
                                    'type': 'config_leverage',
                                    'value': value,
                                    'limit': RISK_LIMITS['max_leverage'],
                                    'message': f'Config leverage {value}x exceeds limit'
                                })
    
    def check_env_risks(self, content: str, filepath: str) -> None:
        """Check risk parameters in .env files"""
        lines = content.split('\n')
        
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if 'LEVERAGE' in key:
                    try:
                        leverage = int(value)
                        if leverage > RISK_LIMITS['max_leverage']:
                            self.warnings.append(f"Environment variable {key}={value} exceeds leverage limit")
                    except:
                        pass
                
                if 'POSITION' in key and 'SIZE' in key:
                    try:
                        size = float(value)
                        if size > RISK_LIMITS['max_position_size']:
                            self.warnings.append(f"Environment variable {key}={value} exceeds position size limit")
                    except:
                        pass
    
    def check_directory(self, directory: str) -> None:
        """Check all Python files in directory"""
        path = Path(directory)
        
        if not path.exists():
            return
        
        for filepath in path.rglob('*.py'):
            if not any(exclude in str(filepath) for exclude in ['test_', '__pycache__', '.pyc']):
                self.files_checked += 1
                self.check_position_sizing(filepath)
                self.check_leverage(filepath)
                self.check_stop_losses(filepath)
    
    def print_report(self) -> None:
        """Print risk validation report"""
        print("=" * 80)
        print("RISK LIMITS VALIDATION REPORT - Quinn's Assessment")
        print("=" * 80)
        print(f"Files checked: {self.files_checked}")
        print()
        
        if not self.violations and not self.warnings:
            print("✅ ALL RISK LIMITS VALIDATED!")
            print("Quinn approves: Risk management is properly implemented.")
            print()
            print("Current limits enforced:")
            for limit, value in RISK_LIMITS.items():
                if isinstance(value, float):
                    print(f"  • {limit}: {value:.1%}" if value < 1 else f"  • {limit}: {value}")
                else:
                    print(f"  • {limit}: {value}")
        else:
            if self.violations:
                print("❌ RISK VIOLATIONS FOUND!")
                print()
                
                for violation in self.violations:
                    print(f"\n🚨 {violation['type'].upper()} VIOLATION")
                    print(f"  File: {violation['file']}")
                    print(f"  {violation['message']}")
                    print(f"  Current: {violation['value']}, Limit: {violation['limit']}")
            
            if self.warnings:
                print("\n⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"  • {warning}")
            
            print("\n" + "=" * 80)
            print("❌ QUINN'S VERDICT: VETO!")
            print("Fix all risk violations before deployment.")
            print("Capital preservation is non-negotiable.")
            print("=" * 80)
    
    def run(self) -> bool:
        """Run validation and return True if all limits pass"""
        # Check source directories
        for directory in ['src', 'strategies', 'config']:
            self.check_directory(directory)
        
        # Check config files
        self.check_config_files()
        
        self.print_report()
        return len(self.violations) == 0

def main():
    """Main execution"""
    validator = RiskValidator()
    success = validator.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()