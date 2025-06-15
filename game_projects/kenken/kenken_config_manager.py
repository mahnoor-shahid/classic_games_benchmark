# kenken_config_manager.py
"""
Configuration Manager for Ken Ken Project
Handles loading, validation, and management of configuration settings
Based on the Sudoku configuration system
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class KenKenConfigManager:
    def __init__(self, config_file: str = "kenken_config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.defaults = self._get_defaults()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for config manager"""
        logger = logging.getLogger('kenken_config_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for Ken Ken"""
        return {
            'project': {
                'name': 'mnist-kenken',
                'version': '1.0.0',
                'data_dir': './data',
                'output_dir': './output',
                'logs_dir': './logs'
            },
            'environment': {
                'python_version': '3.8+',
                'create_virtual_env': True,
                'venv_name': 'mnist-kenken-env',
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
                    'easy': 10,
                    'moderate': 8,
                    'hard': 5
                },
                'grid_sizes': {
                    'easy': [4, 5],
                    'moderate': 6,
                    'hard': 7
                },
                'complexity': {
                    'easy': {
                        'max_cage_size': 3,
                        'operations': ['add', 'subtract'],
                        'min_cages': 3,
                        'max_cages': 6,
                        'prefer_simple_operations': True
                    },
                    'moderate': {
                        'max_cage_size': 4,
                        'operations': ['add', 'subtract', 'multiply'],
                        'min_cages': 6,
                        'max_cages': 12,
                        'allow_complex_shapes': True
                    },
                    'hard': {
                        'max_cage_size': 6,
                        'operations': ['add', 'subtract', 'multiply', 'divide'],
                        'min_cages': 8,
                        'max_cages': 18,
                        'require_advanced_operations': True,
                        'allow_complex_constraints': True
                    }
                },
                'strategies': {
                    'include_all_easy': True,
                    'include_all_moderate': True,
                    'include_all_hard': True,
                    'custom_strategy_weights': {
                        'single_cell_cage': 1.0,
                        'simple_arithmetic': 1.0,
                        'cage_elimination': 0.8,
                        'advanced_combinations': 0.6,
                        'constraint_propagation': 0.4
                    }
                },
                'mnist': {
                    'image_size': 28,
                    'cell_size': 64,  # Larger cells for better visibility
                    'use_augmentation': False,
                    'brightness_variation': 0.1,
                    'rotation_angle': 0,
                    'noise_level': 0.0,
                    'cage_boundary_thickness': 6,
                    'cage_corner_size': 12,
                    'operation_label_size': 12
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
                    'cage_analysis': True,
                    'operation_distribution': True,
                    'solving_time_analysis': True,
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
                    'image_quality': 95,
                    'save_cage_overlays': True
                },
                'documentation': {
                    'generate_dataset_report': True,
                    'include_strategy_documentation': True,
                    'create_usage_examples': True,
                    'export_operation_rules': True
                }
            },
            'validation': {
                'strict_mode': True,
                'check_uniqueness': True,
                'verify_solvability': True,
                'validate_strategies': True,
                'max_solve_time_seconds': 45,
                'test_sample_size': 5,
                'quality': {
                    'min_puzzle_difficulty_score': 0.3,
                    'max_puzzle_difficulty_score': 0.9,
                    'require_human_solvable': True,
                    'avoid_guessing': True,
                    'validate_cage_constraints': True
                }
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'file_output': True,
                'log_file': 'mnist_kenken.log',
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
            
            # Validate grid sizes
            grid_sizes = generation.get('grid_sizes', {})
            for difficulty, sizes in grid_sizes.items():
                if not isinstance(sizes, list) or not all(isinstance(s, int) and 3 <= s <= 9 for s in sizes):
                    self.logger.error(f"Invalid grid sizes for {difficulty}: {sizes}")
                    return False
                
            # In _validate_config method, find the grid sizes validation and replace with:
            grid_sizes = generation.get('grid_sizes', {})
            for difficulty, sizes in grid_sizes.items():
                # Allow both single integers and lists
                if isinstance(sizes, int):
                    if not (3 <= sizes <= 9):
                        self.logger.error(f"Invalid grid size for {difficulty}: {sizes}")
                        return False
                elif isinstance(sizes, list):
                    if not all(isinstance(s, int) and 3 <= s <= 9 for s in sizes):
                        self.logger.error(f"Invalid grid sizes for {difficulty}: {sizes}")
                        return False
                else:
                    self.logger.error(f"Invalid grid sizes format for {difficulty}: {sizes}")
                    return False
                        
            # Validate complexity settings
            complexity = generation.get('complexity', {})
            for difficulty, settings in complexity.items():
                max_cage_size = settings.get('max_cage_size', 0)
                operations = settings.get('operations', [])
                
                if max_cage_size < 2 or max_cage_size > 8:
                    self.logger.error(f"Invalid max cage size for {difficulty}: {max_cage_size}")
                    return False
                
                valid_operations = {'add', 'subtract', 'multiply', 'divide'}
                if not all(op in valid_operations for op in operations):
                    self.logger.error(f"Invalid operations for {difficulty}: {operations}")
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during configuration validation: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
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
            'easy': self.get('generation.num_puzzles.easy', 10),
            'moderate': self.get('generation.num_puzzles.moderate', 8),
            'hard': self.get('generation.num_puzzles.hard', 5)
        }
    
    def get_grid_sizes(self, difficulty: str):
        """Get grid size(s) for a specific difficulty - returns list for easy, int for others"""
        if difficulty == 'easy':
            return self.get(f'generation.grid_sizes.{difficulty}', [4, 5])
        else:
            defaults = {'moderate': 6, 'hard': 7}
            return self.get(f'generation.grid_sizes.{difficulty}', defaults.get(difficulty, 6))
    
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
        logger = logging.getLogger('mnist_kenken')
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
                log_config.get('log_file', 'mnist_kenken.log')
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
    
    def print_summary(self):
        """Print a summary of current configuration"""
        print("\n" + "=" * 60)
        print("📋 KEN KEN CONFIGURATION SUMMARY")
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
            grid_sizes = self.get_grid_sizes(difficulty)
            print(f"  {difficulty.capitalize()}: {count} puzzles, grid sizes: {grid_sizes}")
        
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
kenken_config_manager = None

def get_kenken_config() -> KenKenConfigManager:
    """Get the global Ken Ken configuration manager instance"""
    global kenken_config_manager
    if kenken_config_manager is None:
        kenken_config_manager = KenKenConfigManager()
        kenken_config_manager.load_config()
    return kenken_config_manager

def load_kenken_config(config_file: str = "kenken_config.yaml") -> bool:
    """Load Ken Ken configuration from file"""
    global kenken_config_manager
    kenken_config_manager = KenKenConfigManager(config_file)
    return kenken_config_manager.load_config()