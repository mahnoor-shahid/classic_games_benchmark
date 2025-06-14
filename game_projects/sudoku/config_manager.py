# config_manager.py
"""
Configuration Manager for MNIST Sudoku Project
Handles loading, validation, and management of configuration settings
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.defaults = self._get_defaults()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for config manager"""
        logger = logging.getLogger('config_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'project': {
                'name': 'mnist-sudoku',
                'version': '1.0.0',
                'data_dir': './data',
                'output_dir': './output',
                'logs_dir': './logs'
            },
            'environment': {
                'python_version': '3.8+',
                'create_virtual_env': True,
                'venv_name': 'mnist-sudoku-env',
                'install_dependencies': True,
                'download_mnist': True
            },
            'hardware': {
                'device': 'auto',
                'num_workers': 4,
                'memory_limit_gb': 8
            },
            'generation': {
                'num_puzzles': {
                    'easy': 100,
                    'moderate': 75,
                    'hard': 50
                },
                'complexity': {
                    'easy': {
                        'min_filled_cells': 35,
                        'max_filled_cells': 45,
                        'max_strategies': 2,
                        'prefer_single_strategy': True
                    },
                    'moderate': {
                        'min_filled_cells': 25,
                        'max_filled_cells': 40,
                        'max_strategies': 3,
                        'allow_composite_strategies': True
                    },
                    'hard': {
                        'min_filled_cells': 17,
                        'max_filled_cells': 30,
                        'max_strategies': 5,
                        'require_advanced_strategies': True
                    }
                },
                'strategies': {
                    'include_all_easy': True,
                    'include_all_moderate': True,
                    'include_all_hard': True,
                    'custom_strategy_weights': {
                        'naked_single': 1.0,
                        'hidden_single_row': 1.0,
                        'naked_pair': 0.8,
                        'x_wing': 0.6,
                        'swordfish': 0.3
                    }
                },
                'mnist': {
                    'image_size': 28,
                    'grid_size': 252,
                    'use_augmentation': False,
                    'brightness_variation': 0.1,
                    'rotation_angle': 0,
                    'noise_level': 0.0
                }
            },
            'analysis': {
                'generate_plots': True,
                'plot_formats': ['png', 'pdf'],
                'plot_dpi': 300,
                'include_statistical_tests': True,
                'validate_all_puzzles': True,
                'visualizations': {
                    'strategy_usage_distribution': True,
                    'difficulty_progression': True,
                    'compositionality_graph': True,
                    'complexity_metrics': True,
                    'solution_time_analysis': True,
                    'mnist_sample_grid': True
                }
            },
            'output': {
                'formats': {
                    'json': True,
                    'csv': True,
                    'pickle': False,
                    'hdf5': False
                },
                'images': {
                    'save_mnist_puzzles': True,
                    'save_solution_images': True,
                    'create_thumbnail_grid': True,
                    'compress_images': True,
                    'image_quality': 95
                },
                'documentation': {
                    'generate_dataset_report': True,
                    'include_strategy_documentation': True,
                    'create_usage_examples': True,
                    'export_fol_rules': True
                }
            },
            'validation': {
                'strict_mode': True,
                'check_uniqueness': True,
                'verify_solvability': True,
                'validate_strategies': True,
                'max_solve_time_seconds': 30,
                'test_sample_size': 5,
                'quality': {
                    'min_puzzle_difficulty_score': 0.3,
                    'max_puzzle_difficulty_score': 0.9,
                    'require_human_solvable': True,
                    'avoid_guessing': True
                }
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'file_output': True,
                'log_file': 'mnist_sudoku.log',
                'max_file_size_mb': 10,
                'backup_count': 3,
                'progress': {
                    'show_progress_bars': True,
                    'update_frequency': 10,
                    'estimated_time': True
                }
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"Config file {self.config_file} not found, creating default...")
                self.create_default_config()
            
            with open(self.config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Merge with defaults to ensure all keys exist
            self.config = self._deep_merge(self.defaults.copy(), loaded_config or {})
            
            # Validate configuration
            if self._validate_config():
                self.logger.info(f"Configuration loaded successfully from {self.config_file}")
                return True
            else:
                self.logger.error("Configuration validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def create_default_config(self):
        """Create default configuration file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.defaults, f, default_flow_style=False, indent=2)
            self.logger.info(f"Default configuration created: {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error creating default config: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate project settings
            project = self.config.get('project', {})
            if not project.get('name'):
                self.logger.error("Project name is required")
                return False
            
            # Validate generation settings
            generation = self.config.get('generation', {})
            num_puzzles = generation.get('num_puzzles', {})
            
            for difficulty in ['easy', 'moderate', 'hard']:
                if difficulty not in num_puzzles:
                    self.logger.error(f"Missing puzzle count for difficulty: {difficulty}")
                    return False
                
                count = num_puzzles[difficulty]
                if not isinstance(count, int) or count <= 0:
                    self.logger.error(f"Invalid puzzle count for {difficulty}: {count}")
                    return False
            
            # Validate complexity settings
            complexity = generation.get('complexity', {})
            for difficulty, settings in complexity.items():
                min_filled = settings.get('min_filled_cells', 0)
                max_filled = settings.get('max_filled_cells', 81)
                
                if min_filled >= max_filled:
                    self.logger.error(f"Invalid filled cells range for {difficulty}: {min_filled}-{max_filled}")
                    return False
                
                if min_filled < 17 or max_filled > 81:
                    self.logger.error(f"Filled cells out of valid range (17-81) for {difficulty}")
                    return False
            
            # Validate hardware settings
            hardware = self.config.get('hardware', {})
            device = hardware.get('device', 'auto')
            if device not in ['auto', 'cpu', 'cuda', 'mps']:
                self.logger.error(f"Invalid device setting: {device}")
                return False
            
            memory_limit = hardware.get('memory_limit_gb', 8)
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                self.logger.error(f"Invalid memory limit: {memory_limit}")
                return False
            
            # Validate logging settings
            logging_config = self.config.get('logging', {})
            log_level = logging_config.get('level', 'INFO')
            if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                self.logger.error(f"Invalid log level: {log_level}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during configuration validation: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'generation.num_puzzles.easy')"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the value
            config_ref[keys[-1]] = value
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config value {key_path}: {e}")
            return False
    
    def save_config(self, filename: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            output_file = filename or self.config_file
            
            with open(output_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create directories specified in configuration"""
        try:
            directories = [
                self.get('project.data_dir'),
                self.get('project.output_dir'),
                self.get('project.logs_dir'),
                os.path.join(self.get('project.output_dir'), 'datasets'),
                os.path.join(self.get('project.output_dir'), 'images'),
                os.path.join(self.get('project.output_dir'), 'analysis'),
                os.path.join(self.get('project.output_dir'), 'reports')
            ]
            
            for directory in directories:
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.debug(f"Created directory: {directory}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            return False
    
    def get_puzzle_counts(self) -> Dict[str, int]:
        """Get puzzle counts for each difficulty"""
        return {
            'easy': self.get('generation.num_puzzles.easy', 50),
            'moderate': self.get('generation.num_puzzles.moderate', 30),
            'hard': self.get('generation.num_puzzles.hard', 20)
        }
    
    def get_complexity_settings(self, difficulty: str) -> Dict[str, Any]:
        """Get complexity settings for a specific difficulty"""
        return self.get(f'generation.complexity.{difficulty}', {})
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights for puzzle generation"""
        return self.get('generation.strategies.custom_strategy_weights', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output format settings"""
        return self.get('output', {})
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """Get analysis settings"""
        return self.get('analysis', {})
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation settings"""
        return self.get('validation', {})
    
    def setup_logging(self) -> logging.Logger:
        """Set up logging based on configuration"""
        log_config = self.get('logging', {})
        
        # Create logger
        logger = logging.getLogger('mnist_sudoku')
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file_output', True):
            log_file = os.path.join(
                self.get('project.logs_dir', './logs'),
                log_config.get('log_file', 'mnist_sudoku.log')
            )
            
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=log_config.get('max_file_size_mb', 10) * 1024 * 1024,
                backupCount=log_config.get('backup_count', 3)
            )
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def export_config(self, format: str = 'yaml', filename: Optional[str] = None) -> bool:
        """Export configuration in different formats"""
        try:
            if not filename:
                base_name = os.path.splitext(self.config_file)[0]
                filename = f"{base_name}_export.{format}"
            
            if format.lower() == 'yaml':
                with open(filename, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(self.config, f, indent=2, default=str)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Configuration exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of current configuration"""
        print("\n" + "=" * 60)
        print("📋 CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Project info
        project = self.get('project', {})
        print(f"Project: {project.get('name')} v{project.get('version')}")
        print(f"Data directory: {project.get('data_dir')}")
        print(f"Output directory: {project.get('output_dir')}")
        
        # Generation settings
        print(f"\n🎯 Generation Settings:")
        puzzle_counts = self.get_puzzle_counts()
        for difficulty, count in puzzle_counts.items():
            print(f"  {difficulty.capitalize()}: {count} puzzles")
        
        # Hardware settings
        hardware = self.get('hardware', {})
        print(f"\n💻 Hardware:")
        print(f"  Device: {hardware.get('device')}")
        print(f"  Workers: {hardware.get('num_workers')}")
        print(f"  Memory limit: {hardware.get('memory_limit_gb')}GB")
        
        # Output settings
        output = self.get('output', {})
        formats = output.get('formats', {})
        enabled_formats = [fmt for fmt, enabled in formats.items() if enabled]
        print(f"\n📄 Output formats: {', '.join(enabled_formats)}")
        
        # Analysis settings
        analysis = self.get('analysis', {})
        if analysis.get('generate_plots'):
            plot_formats = analysis.get('plot_formats', [])
            print(f"📊 Plot formats: {', '.join(plot_formats)}")
        
        print("=" * 60)

# Global config manager instance
config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
        config_manager.load_config()
    return config_manager

def load_config(config_file: str = "config.yaml") -> bool:
    """Load configuration from file"""
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager.load_config()