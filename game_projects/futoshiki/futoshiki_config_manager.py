# futoshiki_config_manager.py
"""
Configuration Manager for MNIST Futoshiki Project
Handles loading, validation, and management of configuration settings
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class FutoshikiConfigManager:
    def __init__(self, config_file: str = "futoshiki_config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.defaults = self._get_defaults()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for config manager"""
        logger = logging.getLogger('futoshiki_config_manager')
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
                'name': 'mnist-futoshiki',
                'version': '2.0.0',
                'data_dir': './data',
                'output_dir': './output',
                'logs_dir': './logs'
            },
            'generation': {
                'puzzle_sizes': {
                    'easy': 4,      # 4x4 for easy
                    'moderate': 5,  # 5x5 for moderate  
                    'hard': 6       # 6x6 for hard
                },
                'num_puzzles': {
                    'easy': 10,
                    'moderate': 8,
                    'hard': 5
                },
                'complexity': {
                    'easy': {
                        'min_filled_cells_ratio': 0.4,
                        'max_filled_cells_ratio': 0.6,
                        'max_strategies': 3,
                        'prefer_single_strategy': True,
                        'constraint_density': 0.3
                    },
                    'moderate': {
                        'min_filled_cells_ratio': 0.25,
                        'max_filled_cells_ratio': 0.45,
                        'max_strategies': 5,
                        'allow_composite_strategies': True,
                        'constraint_density': 0.4
                    },
                    'hard': {
                        'min_filled_cells_ratio': 0.15,
                        'max_filled_cells_ratio': 0.35,
                        'max_strategies': 7,
                        'require_advanced_strategies': True,
                        'constraint_density': 0.5
                    }
                },
                'mnist': {
                    'image_size': 28,
                    'use_same_digits_for_puzzle_and_solution': True,
                    'use_augmentation': False,
                    'brightness_variation': 0.0,
                    'rotation_angle': 0,
                    'noise_level': 0.0
                },
                'constraints': {
                    'visualization': {
                        'show_inequality_symbols': True,
                        'symbol_size': 16,
                        'symbol_color': 'red',
                        'symbol_font': 'Arial',
                        'symbol_position': 'between_cells'
                    },
                    'generation': {
                        'min_constraints_per_puzzle': 3,
                        'max_constraints_per_puzzle': 15,
                        'prefer_constraint_chains': True,
                        'avoid_isolated_constraints': True
                    }
                }
            },
            'output': {
                'formats': {
                    'json': True,
                    'csv': True
                },
                'images': {
                    'save_mnist_puzzles': True,
                    'save_solution_images': True,
                    'save_constraint_overlay': True,
                    'include_grid_lines': True,
                    'cell_border_width': 2,
                    'compress_images': True,
                    'image_quality': 95
                }
            },
            'validation': {
                'strict_mode': True,
                'check_uniqueness': True,
                'verify_solvability': True,
                'validate_strategies': True,
                'max_solve_time_seconds': 60,
                'check_mnist_consistency': True,
                'quality': {
                    'min_puzzle_difficulty_score': 0.3,
                    'max_puzzle_difficulty_score': 0.9,
                    'require_human_solvable': True,
                    'avoid_guessing': True,
                    'ensure_constraint_consistency': True
                }
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'file_output': True,
                'log_file': 'mnist_futoshiki.log'
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
            
            # Validate puzzle sizes
            puzzle_sizes = self.config.get('generation', {}).get('puzzle_sizes', {})
            for difficulty, size in puzzle_sizes.items():
                if not isinstance(size, int) or size < 3 or size > 9:
                    self.logger.error(f"Invalid puzzle size for {difficulty}: {size} (must be 3-9)")
                    return False
            
            # Validate puzzle counts
            num_puzzles = self.config.get('generation', {}).get('num_puzzles', {})
            for difficulty, count in num_puzzles.items():
                if not isinstance(count, int) or count <= 0:
                    self.logger.error(f"Invalid puzzle count for {difficulty}: {count}")
                    return False
            
            # Validate complexity settings
            complexity = self.config.get('generation', {}).get('complexity', {})
            for difficulty, settings in complexity.items():
                min_ratio = settings.get('min_filled_cells_ratio', 0)
                max_ratio = settings.get('max_filled_cells_ratio', 1)
                
                if min_ratio >= max_ratio or min_ratio < 0 or max_ratio > 1:
                    self.logger.error(f"Invalid filled cells ratio for {difficulty}: {min_ratio}-{max_ratio}")
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
    
    def get_puzzle_size(self, difficulty: str) -> int:
        """Get puzzle size for a specific difficulty"""
        return self.get(f'generation.puzzle_sizes.{difficulty}', 4)
    
    def get_puzzle_count(self, difficulty: str) -> int:
        """Get puzzle count for a specific difficulty"""
        return self.get(f'generation.num_puzzles.{difficulty}', 5)
    
    def get_puzzle_counts(self) -> Dict[str, int]:
        """Get puzzle counts for all difficulties"""
        return {
            'easy': self.get_puzzle_count('easy'),
            'moderate': self.get_puzzle_count('moderate'),
            'hard': self.get_puzzle_count('hard')
        }
    
    def get_puzzle_sizes(self) -> Dict[str, int]:
        """Get puzzle sizes for all difficulties"""
        return {
            'easy': self.get_puzzle_size('easy'),
            'moderate': self.get_puzzle_size('moderate'),
            'hard': self.get_puzzle_size('hard')
        }
    
    def get_complexity_settings(self, difficulty: str) -> Dict[str, Any]:
        """Get complexity settings for a specific difficulty"""
        return self.get(f'generation.complexity.{difficulty}', {})
    
    def should_use_same_mnist_digits(self) -> bool:
        """Check if same MNIST digits should be used for puzzle and solution"""
        return self.get('generation.mnist.use_same_digits_for_puzzle_and_solution', True)
    
    def should_show_inequality_symbols(self) -> bool:
        """Check if inequality symbols should be shown"""
        return self.get('generation.constraints.visualization.show_inequality_symbols', True)
    
    def get_constraint_visualization_settings(self) -> Dict[str, Any]:
        """Get constraint visualization settings"""
        return self.get('generation.constraints.visualization', {})
    
    def should_save_constraint_overlay(self) -> bool:
        """Check if constraint overlay should be saved"""
        return self.get('output.images.save_constraint_overlay', True)
    
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of current configuration"""
        print("\n" + "=" * 60)
        print("📋 FUTOSHIKI CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Project info
        project = self.get('project', {})
        print(f"Project: {project.get('name')} v{project.get('version')}")
        print(f"Data directory: {project.get('data_dir')}")
        print(f"Output directory: {project.get('output_dir')}")
        
        # Puzzle settings
        print(f"\n🎯 Puzzle Generation Settings:")
        puzzle_sizes = self.get_puzzle_sizes()
        puzzle_counts = self.get_puzzle_counts()
        
        for difficulty in ['easy', 'moderate', 'hard']:
            size = puzzle_sizes[difficulty]
            count = puzzle_counts[difficulty]
            print(f"  {difficulty.capitalize()}: {count} puzzles, {size}x{size} grid")
        
        # MNIST settings
        print(f"\n🖼️ MNIST Settings:")
        mnist_settings = self.get('generation.mnist', {})
        print(f"  Same digits for puzzle/solution: {mnist_settings.get('use_same_digits_for_puzzle_and_solution', True)}")
        print(f"  Image size: {mnist_settings.get('image_size', 28)}x{mnist_settings.get('image_size', 28)}")
        
        # Constraint visualization
        print(f"\n⚖️ Constraint Visualization:")
        show_symbols = self.should_show_inequality_symbols()
        print(f"  Show inequality symbols: {show_symbols}")
        if show_symbols:
            viz_settings = self.get_constraint_visualization_settings()
            print(f"  Symbol size: {viz_settings.get('symbol_size', 16)}")
            print(f"  Symbol color: {viz_settings.get('symbol_color', 'red')}")
        
        print("=" * 60)


# Global config manager instance
config_manager = None

def get_config() -> FutoshikiConfigManager:
    """Get the global configuration manager instance"""
    global config_manager
    if config_manager is None:
        config_manager = FutoshikiConfigManager()
        config_manager.load_config()
    return config_manager

def load_config(config_file: str = "futoshiki_config.yaml") -> bool:
    """Load configuration from file"""
    global config_manager
    config_manager = FutoshikiConfigManager(config_file)
    return config_manager.load_config()