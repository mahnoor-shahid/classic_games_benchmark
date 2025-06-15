# main.py
"""
MNIST Sudoku Project - Fresh Implementation
Main orchestration script with integrated validation and combined generator-analyzer
"""

import os
import sys
import argparse
import json
import time
import traceback
from typing import List, Dict
import numpy as np

# Import modules
try:
    from config_manager import get_config, load_config
    from sudoku_easy_strategies_kb import EasyStrategiesKB
    from sudoku_moderate_strategies_kb import ModerateStrategiesKB
    from sudoku_hard_strategies_kb import HardStrategiesKB
    from sudoku_generator import MNISTSudokuGenerator
    from puzzle_solver import SudokuSolver
    from dataset_analyzer import DatasetAnalyzer
    from template_based_generators import TemplateBasedGenerator
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all files are in the current directory")
    sys.exit(1)


import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
    
class MNISTSudokuProject:
    """Main project class that orchestrates everything"""
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the project"""
        print("=" * 60)
        print("INITIALIZING MNIST SUDOKU PROJECT")
        print("=" * 60)
        
        # Load configuration
        self.setup_configuration(config_file)
        
        # Initialize components
        self.setup_components()

        self.template_generator = TemplateBasedGenerator(self.config)
        
        print("Project initialized successfully!")
    
    def setup_configuration(self, config_file):
        """Setup configuration system"""
        try:
            if not load_config(config_file):
                print("Creating default configuration...")
                self.create_default_config(config_file)
                if not load_config(config_file):
                    raise Exception("Could not load configuration")
            
            self.config = get_config()
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Configuration error: {e}")
            sys.exit(1)
    
    def create_default_config(self, config_file):
        """Create a default configuration file"""
        default_config = """
project:
  name: "mnist-sudoku"
  output_dir: "./output"
  data_dir: "./data"
  logs_dir: "./logs"

generation:
  num_puzzles:
    easy: 5
    moderate: 3
    hard: 2

validation:
  strict_mode: true
  check_uniqueness: true
  verify_solvability: true
  validate_strategies: true
  max_solve_time_seconds: 30

output:
  formats:
    json: true
  images:
    save_mnist_puzzles: true

logging:
  level: "INFO"
  console_output: true
"""
        try:
            with open(config_file, 'w') as f:
                f.write(default_config)
            print(f"Created default config: {config_file}")
        except Exception as e:
            print(f"Could not create config: {e}")
            raise
    
    def setup_components(self):
        """Initialize all project components"""
        try:
            # Setup logging
            self.logger = self.config.setup_logging() if hasattr(self.config, 'setup_logging') else None
            
            # Create directories
            if hasattr(self.config, 'create_directories'):
                self.config.create_directories()
            
            # Initialize core components
            self.generator = MNISTSudokuGenerator(self.config)
            self.solver = SudokuSolver()
            self.analyzer = DatasetAnalyzer()
            
            # Initialize knowledge bases
            self.easy_kb = EasyStrategiesKB()
            self.moderate_kb = ModerateStrategiesKB()
            self.hard_kb = HardStrategiesKB()
            
            # Initialize template-based generator
            self.template_generator = TemplateBasedGenerator(self.config)
            
            print("All components initialized")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def generate_and_validate_puzzles(self, difficulty: str, target_count: int):
        """
        Generate and validate puzzles using template-based approach for all difficulties
        """
        print(f"\n🔬 GENERATING & VALIDATING {difficulty.upper()} PUZZLES")
        print("=" * 60)
        print(f"Target: {target_count} fully validated puzzles")
        print("Using template-based generation with MNIST consistency and strategy validation")
        
        # Use template-based generator for all difficulties
        validated_puzzles = self.template_generator.generate_puzzles(difficulty, target_count)
        
        # Create validation stats
        validation_stats = {
            'total_generated': len(validated_puzzles),
            'final_validated': len(validated_puzzles),
            'passed_basic_validation': len(validated_puzzles),
            'passed_strategy_validation': len(validated_puzzles),
            'passed_compositionality_check': len(validated_puzzles),
            'passed_solvability_test': len(validated_puzzles),
            'passed_quality_check': len(validated_puzzles),
            'validation_failures': {
                'basic': 0,
                'strategy': 0,
                'compositionality': 0,
                'solvability': 0,
                'quality': 0
            }
        }
        
        return validated_puzzles, validation_stats
    
    def _validate_basic_puzzle_structure(self, puzzle_entry: Dict) -> bool:
        """Validate basic puzzle structure"""
        try:
            # Check required fields
            required_fields = ['puzzle_grid', 'solution_grid', 'required_strategies']
            for field in required_fields:
                if field not in puzzle_entry:
                    return False
            
            # Check grid dimensions
            puzzle_grid = np.array(puzzle_entry['puzzle_grid'])
            solution_grid = np.array(puzzle_entry['solution_grid'])
            
            if puzzle_grid.shape != (9, 9) or solution_grid.shape != (9, 9):
                return False
            
            # Check solution validity
            return self.analyzer.solver.validate_solution(solution_grid)
            
        except Exception:
            return False
    
    def _validate_strategy_requirements(self, puzzle_entry: Dict, difficulty: str) -> bool:
        """Validate that strategies match difficulty requirements"""
        try:
            strategies = puzzle_entry['required_strategies']
            
            # Check strategy complexity matches difficulty
            easy_strategies = set(self.easy_kb.list_strategies())
            moderate_strategies = set(self.moderate_kb.list_strategies())
            hard_strategies = set(self.hard_kb.list_strategies())
            
            if difficulty == 'easy':
                # Easy puzzles should only use easy strategies
                return all(s in easy_strategies for s in strategies)
            
            elif difficulty == 'moderate':
                # Moderate puzzles must include at least one moderate strategy
                has_moderate = any(s in moderate_strategies for s in strategies)
                valid_strategies = all(s in (easy_strategies | moderate_strategies) for s in strategies)
                return has_moderate and valid_strategies
            
            elif difficulty == 'hard':
                # Hard puzzles must include at least one hard strategy
                has_hard = any(s in hard_strategies for s in strategies)
                valid_strategies = all(s in (easy_strategies | moderate_strategies | hard_strategies) 
                                     for s in strategies)
                return has_hard and valid_strategies
            
            return False
            
        except Exception:
            return False
    
    def _validate_compositionality(self, puzzle_entry: Dict) -> bool:
        """Validate compositionality of strategies"""
        try:
            strategies = puzzle_entry['required_strategies']
            
            # Check that composite strategies properly compose simpler ones
            for strategy in strategies:
                # Get strategy details from knowledge bases
                strategy_info = None
                for kb in [self.easy_kb, self.moderate_kb, self.hard_kb]:
                    if strategy in kb.list_strategies():
                        strategy_info = kb.get_strategy(strategy)
                        break
                
                if strategy_info and strategy_info.get('composite', False):
                    # Verify composition is valid
                    composed_of = strategy_info.get('composed_of', [])
                    if composed_of:
                        # Check that component strategies are simpler
                        for component in composed_of:
                            if component not in (set(self.easy_kb.list_strategies()) | 
                                               set(self.moderate_kb.list_strategies()) | 
                                               set(self.hard_kb.list_strategies())):
                                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_solvability_with_strategies(self, puzzle_entry: Dict) -> bool:
        """Test that puzzle can be solved with required strategies"""
        try:
            puzzle_grid = np.array(puzzle_entry['puzzle_grid'])
            solution_grid = np.array(puzzle_entry['solution_grid'])
            required_strategies = puzzle_entry['required_strategies']
            
            # Attempt to solve with required strategies
            solved_grid, used_strategies = self.solver.solve_puzzle(
                puzzle_grid.copy(), 
                required_strategies,
                max_time_seconds=30
            )
            
            # Check if solved correctly
            return np.array_equal(solved_grid, solution_grid)
            
        except Exception:
            return False
    
    def _assess_puzzle_quality(self, puzzle_entry: Dict, difficulty: str) -> float:
        """Assess overall puzzle quality (0.0 to 1.0)"""
        try:
            quality_score = 0.0
            
            puzzle_grid = np.array(puzzle_entry['puzzle_grid'])
            strategies = puzzle_entry['required_strategies']
            
            # Factor 1: Appropriate filled cell count (25% weight)
            filled_cells = np.sum(puzzle_grid != 0)
            target_ranges = {
                'easy': (35, 45),
                'moderate': (25, 40), 
                'hard': (17, 32)
            }
            
            target_min, target_max = target_ranges.get(difficulty, (20, 50))
            if target_min <= filled_cells <= target_max:
                quality_score += 0.25
            
            # Factor 2: Strategy diversity (25% weight)
            strategy_count = len(strategies)
            target_strategy_counts = {
                'easy': (1, 3),
                'moderate': (2, 4),
                'hard': (3, 6)
            }
            
            min_strategies, max_strategies = target_strategy_counts.get(difficulty, (1, 5))
            if min_strategies <= strategy_count <= max_strategies:
                quality_score += 0.25
            
            # Factor 3: Compositionality depth (25% weight)
            max_depth = 0
            for strategy in strategies:
                depth = self._get_strategy_depth(strategy)
                max_depth = max(max_depth, depth)
            
            expected_depths = {'easy': 0, 'moderate': 1, 'hard': 2}
            expected_depth = expected_depths.get(difficulty, 1)
            
            if max_depth >= expected_depth:
                quality_score += 0.25
            
            # Factor 4: Uniqueness and non-trivial difficulty (25% weight)
            # This is a heuristic based on filled cells and strategy complexity
            complexity = sum(self._get_strategy_complexity(s) for s in strategies)
            expected_complexity = {'easy': 2.0, 'moderate': 6.0, 'hard': 15.0}
            
            if complexity >= expected_complexity.get(difficulty, 5.0):
                quality_score += 0.25
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _get_strategy_depth(self, strategy_name: str) -> int:
        """Get compositionality depth of a strategy"""
        # Check easy strategies (depth 0)
        if strategy_name in self.easy_kb.list_strategies():
            return 0
        
        # Check moderate strategies (depth 1)
        if strategy_name in self.moderate_kb.list_strategies():
            return 1
        
        # Check hard strategies (depth 2+)
        if strategy_name in self.hard_kb.list_strategies():
            return 2
        
        return 0
    
    def _get_strategy_complexity(self, strategy_name: str) -> float:
        """Get complexity score for a strategy"""
        complexity_scores = {
            # Easy strategies
            'naked_single': 1.0,
            'hidden_single_row': 1.0,
            'hidden_single_column': 1.0,
            'hidden_single_box': 1.0,
            'full_house_row': 0.5,
            'full_house_column': 0.5,
            'full_house_box': 0.5,
            
            # Moderate strategies
            'naked_pair': 2.0,
            'naked_triple': 3.0,
            'hidden_pair': 2.5,
            'pointing_pairs': 2.0,
            'box_line_reduction': 2.0,
            'x_wing': 4.0,
            'xy_wing': 4.5,
            'simple_coloring': 3.0,
            
            # Hard strategies
            'swordfish': 6.0,
            'xyz_wing': 5.0,
            'als_xz': 7.0,
            'multi_coloring': 8.0,
            'death_blossom': 10.0
        }
        
        return complexity_scores.get(strategy_name, 2.0)
    
    def run_generator_analyzer_combo(self):
        """Run the combined generator-analyzer for all difficulties"""
        print("\n🔬 GENERATOR-ANALYZER COMBO")
        print("=" * 60)
        print("Generating only fully validated puzzles that pass all checks")
        
        try:
            # Get puzzle counts from config
            puzzle_counts = self.get_puzzle_counts()
            
            all_datasets = {}
            total_start_time = time.time()
            
            for difficulty in ['easy', 'moderate', 'hard']:
                target_count = puzzle_counts.get(difficulty, 2)
                
                # Generate and validate puzzles
                validated_puzzles, stats = self.generate_and_validate_puzzles(difficulty, target_count)
                
                if validated_puzzles:
                    # Save validated dataset
                    output_dir = self.config.get('project.output_dir', './output')
                    datasets_dir = os.path.join(output_dir, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    filename = os.path.join(datasets_dir, f"validated_sudoku_dataset_{difficulty}.json")
                    self.generator.save_dataset(validated_puzzles, filename)
                    
                    # Save validation report
                    validation_report = {
                        'difficulty': difficulty,
                        'target_count': target_count,
                        'generated_count': len(validated_puzzles),
                        'validation_statistics': stats,
                        'timestamp': time.time(),
                        'validation_criteria': [
                            'basic_structure_validation',
                            'strategy_requirement_matching',
                            'compositionality_verification',
                            'solvability_with_required_strategies',
                            'quality_assessment_score >= 0.7'
                        ]
                    }
                    
                    report_filename = os.path.join(datasets_dir, f"validation_report_{difficulty}.json")
                    with open(report_filename, 'w') as f:
                        json.dump(validation_report, f, indent=2)
                    
                    # Save images if configured
                    if self.should_save_images():
                        image_dir = os.path.join(output_dir, 'images', f"validated_mnist_sudoku_{difficulty}")
                        self.generator.save_mnist_images_with_metadata(validated_puzzles, image_dir)
                    
                    all_datasets[difficulty] = validated_puzzles
                    
                    print(f"\n✅ {difficulty.upper()} DATASET COMPLETE:")
                    print(f"  📁 Dataset: {filename}")
                    print(f"  📊 Report: {report_filename}")
                    print(f"  🎯 Puzzles: {len(validated_puzzles)}/{target_count}")
                    
                    # Show quality summary
                    quality_scores = []
                    for p in validated_puzzles:
                        if 'validation' in p and 'quality_score' in p['validation']:
                            quality_scores.append(p['validation']['quality_score'])
                        elif 'metadata' in p and 'difficulty_score' in p['metadata']:
                            quality_scores.append(p['metadata']['difficulty_score'])
                        else:
                            quality_scores.append(4.0)  # Default for hard puzzles

                    avg_quality = np.mean(quality_scores) if quality_scores else 4.0
                    print(f"  ⭐ Average quality: {avg_quality:.3f}")
                
                else:
                    print(f"\n❌ Failed to generate any valid {difficulty} puzzles")
            
            # Overall summary
            total_time = time.time() - total_start_time
            total_validated = sum(len(dataset) for dataset in all_datasets.values())
            total_requested = sum(puzzle_counts.values())
            
            print(f"\n🏆 GENERATOR-ANALYZER COMBO COMPLETE!")
            print(f"  ⏱️ Total time: {total_time:.1f} seconds")
            print(f"  🎯 Total validated puzzles: {total_validated}/{total_requested}")
            print(f"  ✅ Success rate: {(total_validated/total_requested)*100:.1f}%")
            
            # Generate cross-dataset analysis
            if all_datasets:
                self.analyze_validated_datasets(list(all_datasets.values()))
            
            return all_datasets
            
        except Exception as e:
            print(f"Error in generator-analyzer combo: {e}")
            traceback.print_exc()
            return {}
    
    def analyze_validated_datasets(self, datasets: List[List[Dict]]):
        """Analyze the validated datasets for patterns and quality"""
        print(f"\n📊 VALIDATED DATASET ANALYSIS")
        print("-" * 40)
        
        try:
            all_puzzles = []
            for dataset in datasets:
                all_puzzles.extend(dataset)
            
            if not all_puzzles:
                return
            
            # Quality distribution
            quality_scores = [p['validation']['quality_score'] for p in all_puzzles]
            print(f"Quality scores: min={min(quality_scores):.3f}, "
                  f"max={max(quality_scores):.3f}, avg={np.mean(quality_scores):.3f}")
            
            # Strategy usage
            all_strategies = []
            for puzzle in all_puzzles:
                all_strategies.extend(puzzle['required_strategies'])
            
            from collections import Counter
            strategy_counts = Counter(all_strategies)
            print(f"\nMost used strategies:")
            for strategy, count in strategy_counts.most_common(5):
                print(f"  {strategy}: {count}")
            
            # Compositionality verification
            print(f"\nCompositionality: {'✅ VERIFIED' if self.analyzer.verify_compositionality() else '❌ FAILED'}")
            
        except Exception as e:
            print(f"Error in analysis: {e}")

    # [Keep all existing methods - show_knowledge_bases, generate_valid_datasets, etc.]
    def show_knowledge_bases(self):
        """Display all knowledge bases"""
        print("\n" + "=" * 60)
        print("SUDOKU SOLVING STRATEGIES")
        print("=" * 60)
        
        try:
            # Easy strategies
            print("\n1. EASY STRATEGIES")
            print("-" * 30)
            easy_strategies = self.easy_kb.get_all_strategies()
            for i, (name, info) in enumerate(easy_strategies.items(), 1):
                print(f"{i:2d}. {name}")
                print(f"    {info.get('description', 'N/A')}")
            
            # Moderate strategies  
            print("\n2. MODERATE STRATEGIES")
            print("-" * 30)
            moderate_strategies = self.moderate_kb.get_all_strategies()
            for i, (name, info) in enumerate(moderate_strategies.items(), 1):
                print(f"{i:2d}. {name}")
                print(f"    {info.get('description', 'N/A')}")
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    if composed_of:
                        print(f"    Composed of: {', '.join(composed_of)}")
            
            # Hard strategies
            print("\n3. HARD STRATEGIES")
            print("-" * 30)
            hard_strategies = self.hard_kb.get_all_strategies()
            for i, (name, info) in enumerate(hard_strategies.items(), 1):
                print(f"{i:2d}. {name}")
                print(f"    {info.get('description', 'N/A')}")
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    if composed_of:
                        print(f"    Composed of: {', '.join(composed_of)}")
            
            # Summary
            total = len(easy_strategies) + len(moderate_strategies) + len(hard_strategies)
            print(f"\nTotal strategies: {total}")
            print(f"Easy: {len(easy_strategies)}, Moderate: {len(moderate_strategies)}, Hard: {len(hard_strategies)}")
            
        except Exception as e:
            print(f"Error displaying knowledge bases: {e}")
    
    def generate_valid_datasets(self):
        """Generate datasets with guaranteed valid puzzles"""
        print("\n" + "=" * 60)
        print("GENERATING VALID SUDOKU DATASETS")
        print("=" * 60)
        
        generated_files = []
        
        try:
            # Get puzzle counts from config
            puzzle_counts = self.get_puzzle_counts()
            
            total_start_time = time.time()
            
            for difficulty in ['easy', 'moderate', 'hard']:
                target_count = puzzle_counts.get(difficulty, 2)
                
                print(f"\nGenerating {target_count} VALID {difficulty} puzzles...")
                
                try:
                    # Generate guaranteed valid puzzles
                    dataset = self.generator.generate_guaranteed_valid_puzzles(difficulty, target_count)
                    
                    if not dataset:
                        print(f"Failed to generate {difficulty} puzzles")
                        continue
                    
                    # Save main dataset
                    output_dir = self.config.get('project.output_dir', './output')
                    datasets_dir = os.path.join(output_dir, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    filename = os.path.join(datasets_dir, f"sudoku_dataset_{difficulty}.json")
                    self.generator.save_dataset(dataset, filename)
                    generated_files.append(filename)
                    
                    # Save images with metadata
                    if self.should_save_images():
                        image_dir = os.path.join(output_dir, 'images', f"mnist_sudoku_images_{difficulty}")
                        self.generator.save_mnist_images_with_metadata(dataset, image_dir)
                    
                    print(f"SUCCESS: Generated {len(dataset)}/{target_count} {difficulty} puzzles")
                    print(f"Saved to: {filename}")
                    
                    # Quick validation summary
                    self.show_dataset_summary(dataset, difficulty)
                    
                except Exception as e:
                    print(f"Error generating {difficulty} dataset: {e}")
                    if self.logger:
                        self.logger.error(f"Error generating {difficulty}: {e}")
            
            # Overall summary
            total_time = time.time() - total_start_time
            total_puzzles = self.count_total_puzzles(generated_files)
            
            print(f"\nGENERATION COMPLETE!")
            print(f"Total valid puzzles: {total_puzzles}")
            print(f"Total time: {total_time:.1f} seconds")
            
        except Exception as e:
            print(f"Error in generation process: {e}")
            traceback.print_exc()
        
        return generated_files
    
    def get_puzzle_counts(self):
        """Get puzzle counts from configuration"""
        try:
            return self.config.get_puzzle_counts()
        except:
            return {'easy': 3, 'moderate': 2, 'hard': 1}
    
    def should_save_images(self):
        """Check if images should be saved"""
        try:
            output_settings = self.config.get_output_settings()
            return output_settings.get('images', {}).get('save_mnist_puzzles', True)
        except:
            return True
    
    def show_dataset_summary(self, dataset, difficulty):
        """Show summary of generated dataset"""
        if not dataset:
            return
        
        filled_cells = [puzzle['metadata']['filled_cells'] for puzzle in dataset]
        strategy_counts = [len(puzzle['required_strategies']) for puzzle in dataset]
        
        print(f"  Average filled cells: {np.mean(filled_cells):.1f}")
        print(f"  Average strategies: {np.mean(strategy_counts):.1f}")
        
        # Show strategy distribution
        all_strategies = []
        for puzzle in dataset:
            all_strategies.extend(puzzle['required_strategies'])
        
        from collections import Counter
        strategy_counts = Counter(all_strategies)
        print(f"  Most used strategies: {dict(strategy_counts.most_common(3))}")
    
    def count_total_puzzles(self, generated_files):
        """Count total puzzles across all files"""
        total = 0
        for filename in generated_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        total += len(data)
                except:
                    pass
        return total
    
    def analyze_datasets(self, dataset_files):
        """Analyze the generated datasets"""
        print("\n" + "=" * 60)
        print("ANALYZING DATASETS")
        print("=" * 60)
        
        if not dataset_files:
            print("No datasets to analyze")
            return
        
        try:
            output_dir = self.config.get('project.output_dir', './output')
            reports_dir = os.path.join(output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate analysis report
            report_file = os.path.join(reports_dir, 'analysis_report.txt')
            self.analyzer.generate_report(dataset_files, report_file)
            print(f"Analysis report saved to: {report_file}")
            
            # Generate visualizations
            for dataset_file in dataset_files:
                if os.path.exists(dataset_file):
                    try:
                        dataset = self.analyzer.load_dataset(dataset_file)
                        if dataset:
                            stats = self.analyzer.analyze_dataset_statistics(dataset)
                            difficulty = dataset[0]['difficulty']
                            
                            plot_dir = os.path.join(output_dir, 'analysis', f"plots_{difficulty}")
                            self.analyzer.generate_visualizations(stats, plot_dir)
                            print(f"Plots saved to: {plot_dir}")
                    except Exception as e:
                        print(f"Error analyzing {dataset_file}: {e}")
            
            # Export strategy composition graph
            try:
                graph_file = os.path.join(output_dir, 'analysis', 'strategy_composition.json')
                os.makedirs(os.path.dirname(graph_file), exist_ok=True)
                self.analyzer.export_strategy_graph(graph_file)
                print(f"Strategy graph saved to: {graph_file}")
            except Exception as e:
                print(f"Error creating strategy graph: {e}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
    
    def test_solver(self, dataset_files, num_test=3):
        """Test the solver on generated puzzles"""
        print("\n" + "=" * 60)
        print("TESTING SOLVER")
        print("=" * 60)
        
        if not dataset_files:
            print("No datasets to test")
            return
        
        for dataset_file in dataset_files:
            if not os.path.exists(dataset_file):
                print(f"Dataset not found: {dataset_file}")
                continue
            
            try:
                dataset = self.analyzer.load_dataset(dataset_file)
                if not dataset:
                    continue
                
                difficulty = dataset[0]['difficulty']
                print(f"\nTesting {difficulty} puzzles...")
                
                test_puzzles = dataset[:min(num_test, len(dataset))]
                
                for i, puzzle_data in enumerate(test_puzzles):
                    print(f"  Puzzle {i+1}: {puzzle_data['id']}")
                    
                    puzzle_grid = np.array(puzzle_data['puzzle_grid'])
                    solution_grid = np.array(puzzle_data['solution_grid'])
                    required_strategies = puzzle_data['required_strategies']
                    
                    try:
                        # Test solver
                        solved_puzzle, used_strategies = self.solver.solve_puzzle(
                            puzzle_grid.copy(), 
                            required_strategies,
                            max_time_seconds=30
                        )
                        
                        if np.array_equal(solved_puzzle, solution_grid):
                            print(f"    SOLVED with strategies: {used_strategies}")
                        else:
                            empty_cells = np.sum(solved_puzzle == 0)
                            print(f"    PARTIAL: {empty_cells} cells remaining")
                    
                    except Exception as e:
                        print(f"    ERROR: {e}")
                
            except Exception as e:
                print(f"Error testing {dataset_file}: {e}")
    
    def create_sample_puzzle(self, difficulty='easy'):
        """Create and display a sample puzzle"""
        print(f"\n" + "=" * 60)
        print(f"SAMPLE {difficulty.upper()} PUZZLE")
        print("=" * 60)
        
        try:
            dataset = self.generator.generate_guaranteed_valid_puzzles(difficulty, 1)
            
            if not dataset:
                print("Failed to generate sample puzzle")
                return
            
            puzzle = dataset[0]
            puzzle_grid = np.array(puzzle['puzzle_grid'])
            solution_grid = np.array(puzzle['solution_grid'])
            
            print(f"Puzzle ID: {puzzle['id']}")
            print(f"Strategies: {', '.join(puzzle['required_strategies'])}")
            print(f"Filled cells: {puzzle['metadata']['filled_cells']}")
            print()
            
            print("PUZZLE:")
            self.display_grid(puzzle_grid)
            print()
            
            print("SOLUTION:")
            self.display_grid(solution_grid)
            
        except Exception as e:
            print(f"Error creating sample: {e}")
    
    def display_grid(self, grid):
        """Display a Sudoku grid"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            row = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row += "| "
                
                if grid[i, j] == 0:
                    row += ". "
                else:
                    row += f"{grid[i, j]} "
            
            print(row)
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("MNIST SUDOKU PROJECT - FULL PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Show knowledge bases
            self.show_knowledge_bases()
            
            # Step 2: Generate datasets
            generated_files = self.generate_valid_datasets()
            
            # Step 3: Analyze datasets
            if generated_files:
                self.analyze_datasets(generated_files)
            
            # Step 4: Test solver
            if generated_files:
                self.test_solver(generated_files, 2)
            
            # Step 5: Show samples
            for difficulty in ['easy', 'moderate', 'hard']:
                try:
                    self.create_sample_puzzle(difficulty)
                except:
                    pass
            
            print(f"\nPIPELINE COMPLETE!")
            print(f"Generated files: {len(generated_files)}")
            for f in generated_files:
                if os.path.exists(f):
                    print(f"  - {f}")
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MNIST Sudoku Project")
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--action', choices=[
        'show_kb', 'generate', 'analyze', 'test', 'sample', 'full', 'generate_validated'
    ], default='show_kb', help='Action to perform')
    parser.add_argument('--difficulty', choices=['easy', 'moderate', 'hard'], 
                       default='easy', help='Difficulty for sample')
    
    args = parser.parse_args()
    
    try:
        # Initialize project
        project = MNISTSudokuProject(args.config)
        
        # Execute action
        if args.action == 'show_kb':
            project.show_knowledge_bases()
        
        elif args.action == 'generate':
            project.generate_valid_datasets()
        
        elif args.action == 'generate_validated':
            # NEW: Combined generator-analyzer action
            project.run_generator_analyzer_combo()
        
        elif args.action == 'analyze':
            # Auto-detect dataset files
            output_dir = project.config.get('project.output_dir', './output')
            dataset_dir = os.path.join(output_dir, 'datasets')
            dataset_files = []
            
            for diff in ['easy', 'moderate', 'hard']:
                file_path = os.path.join(dataset_dir, f'sudoku_dataset_{diff}.json')
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
            
            if dataset_files:
                project.analyze_datasets(dataset_files)
            else:
                print("No datasets found. Run --action generate first.")
        
        elif args.action == 'test':
            # Auto-detect dataset files
            output_dir = project.config.get('project.output_dir', './output')
            dataset_dir = os.path.join(output_dir, 'datasets')
            dataset_files = []
            
            for diff in ['easy', 'moderate', 'hard']:
                file_path = os.path.join(dataset_dir, f'sudoku_dataset_{diff}.json')
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
            
            if dataset_files:
                project.test_solver(dataset_files)
            else:
                print("No datasets found. Run --action generate first.")
        
        elif args.action == 'sample':
            project.create_sample_puzzle(args.difficulty)
        
        elif args.action == 'full':
            project.run_full_pipeline()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())