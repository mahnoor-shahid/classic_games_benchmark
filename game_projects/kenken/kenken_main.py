# kenken_main.py
"""
MNIST Ken Ken Project - Main Orchestration Script
Similar structure to the Sudoku main.py with Ken Ken-specific adaptations
"""

import os
import sys
import argparse
import json
import time
import traceback
from typing import List, Dict
import numpy as np

# Import Ken Ken modules
try:
    from kenken_config_manager import get_kenken_config, load_kenken_config
    from kenken_easy_strategies_kb import KenKenEasyStrategiesKB
    from kenken_moderate_strategies_kb import KenKenModerateStrategiesKB
    from kenken_hard_strategies_kb import KenKenHardStrategiesKB
    from kenken_generator import MNISTKenKenGenerator
    from kenken_solver import KenKenSolver
    from kenken_template_generators import KenKenTemplateBasedGenerator
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all Ken Ken files are in the current directory")
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
    
class MNISTKenKenProject:
    """Main Ken Ken project class that orchestrates everything"""
    
    def __init__(self, config_file="kenken_config.yaml"):
        """Initialize the Ken Ken project"""
        print("=" * 60)
        print("INITIALIZING MNIST KEN KEN PROJECT")
        print("=" * 60)
        
        # Load configuration
        self.setup_configuration(config_file)
        
        # Initialize components
        self.setup_components()
        
        print("Ken Ken Project initialized successfully!")
    
    def setup_configuration(self, config_file):
        """Setup configuration system"""
        try:
            if not load_kenken_config(config_file):
                print("Creating default Ken Ken configuration...")
                self.create_default_config(config_file)
                if not load_kenken_config(config_file):
                    raise Exception("Could not load configuration")
            
            self.config = get_kenken_config()
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Configuration error: {e}")
            sys.exit(1)
    
    def create_default_config(self, config_file):
        """Create a default configuration file"""
        default_config = """
project:
  name: "mnist-kenken"
  output_dir: "./output"
  data_dir: "./data"
  logs_dir: "./logs"

generation:
  num_puzzles:
    easy: 5
    moderate: 3
    hard: 2
  grid_sizes:
    easy: [4, 5]
    moderate: [5, 6] 
    hard: [6, 7]

validation:
  strict_mode: true
  check_uniqueness: true
  verify_solvability: true
  validate_strategies: true
  max_solve_time_seconds: 45

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
            self.generator = MNISTKenKenGenerator(self.config)
            self.solver = KenKenSolver()
            
            # Initialize knowledge bases
            self.easy_kb = KenKenEasyStrategiesKB()
            self.moderate_kb = KenKenModerateStrategiesKB()
            self.hard_kb = KenKenHardStrategiesKB()
            
            # Initialize template-based generator
            self.template_generator = KenKenTemplateBasedGenerator(self.config)
            
            print("All Ken Ken components initialized")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def show_knowledge_bases(self):
        """Display all Ken Ken knowledge bases"""
        print("\n" + "=" * 60)
        print("KEN KEN SOLVING STRATEGIES")
        print("=" * 60)
        
        try:
            # Easy strategies
            print("\n1. EASY STRATEGIES")
            print("-" * 30)
            easy_strategies = self.easy_kb.get_all_strategies()
            for i, (name, info) in enumerate(easy_strategies.items(), 1):
                print(f"{i:2d}. {name}")
                print(f"    {info.get('description', 'N/A')}")
                operations = self.easy_kb.get_operations_used(name)
                if operations:
                    print(f"    Operations: {', '.join(operations)}")
            
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
                operations = self.moderate_kb.get_operations_used(name)
                if operations:
                    print(f"    Operations: {', '.join(operations)}")
            
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
                operations = self.hard_kb.get_operations_used(name)
                if operations:
                    print(f"    Operations: {', '.join(operations)}")
            
            # Summary
            total = len(easy_strategies) + len(moderate_strategies) + len(hard_strategies)
            print(f"\nTotal Ken Ken strategies: {total}")
            print(f"Easy: {len(easy_strategies)}, Moderate: {len(moderate_strategies)}, Hard: {len(hard_strategies)}")
            
            # Show compositionality structure
            print(f"\n🔗 COMPOSITIONALITY STRUCTURE:")
            print(f"  Easy strategies form the foundation")
            print(f"  Moderate strategies compose easy strategies")
            print(f"  Hard strategies compose easy + moderate strategies")
            
        except Exception as e:
            print(f"Error displaying knowledge bases: {e}")
    
    def generate_valid_datasets(self):
        """Generate datasets with guaranteed valid Ken Ken puzzles"""
        print("\n" + "=" * 60)
        print("GENERATING VALID KEN KEN DATASETS")
        print("=" * 60)
        
        generated_files = []
        
        try:
            # Get puzzle counts from config
            puzzle_counts = self.get_puzzle_counts()
            
            total_start_time = time.time()
            
            for difficulty in ['easy', 'moderate', 'hard']:
                target_count = puzzle_counts.get(difficulty, 2)
                grid_sizes = self.get_grid_sizes_for_difficulty(difficulty)
                
                if difficulty == 'easy':
                    print(f"\nGenerating {target_count} VALID {difficulty} Ken Ken puzzles...")
                    print(f"Grid sizes: {grid_sizes} (mix of 4x4 and 5x5)")
                else:
                    print(f"\nGenerating {target_count} VALID {difficulty} Ken Ken puzzles...")
                    print(f"Grid size: {grid_sizes}x{grid_sizes}")
                
                try:
                    # Generate guaranteed valid puzzles
                    dataset = self.generator.generate_guaranteed_valid_puzzles(
                        difficulty, target_count, grid_size=None
                    )
                    
                    if not dataset:
                        print(f"Failed to generate {difficulty} puzzles")
                        continue
                    
                    # Save main dataset
                    output_dir = self.config.get('project.output_dir', './output')
                    datasets_dir = os.path.join(output_dir, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    filename = os.path.join(datasets_dir, f"kenken_dataset_{difficulty}.json")
                    self.generator.save_dataset(dataset, filename)
                    generated_files.append(filename)
                    
                    # Save images with metadata
                    if self.should_save_images():
                        image_dir = os.path.join(output_dir, 'images', f"mnist_kenken_images_{difficulty}")
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
            
            print(f"\nKEN KEN GENERATION COMPLETE!")
            print(f"Total valid puzzles: {total_puzzles}")
            print(f"Total time: {total_time:.1f} seconds")
            
        except Exception as e:
            print(f"Error in generation process: {e}")
            traceback.print_exc()
        
        return generated_files
    
    def run_generator_analyzer_combo(self):
        """Run the combined generator-analyzer for all difficulties using templates"""
        print("\n🔬 KEN KEN GENERATOR-ANALYZER COMBO")
        print("=" * 60)
        print("Generating only fully validated Ken Ken puzzles that pass all checks")
        
        try:
            # Get puzzle counts from config
            puzzle_counts = self.get_puzzle_counts()
            
            all_datasets = {}
            total_start_time = time.time()
            
            for difficulty in ['easy', 'moderate', 'hard']:
                target_count = puzzle_counts.get(difficulty, 2)
                
                # Generate and validate puzzles using templates
                validated_puzzles = self.template_generator.generate_puzzles(difficulty, target_count)
                
                if validated_puzzles:
                    # Save validated dataset
                    output_dir = self.config.get('project.output_dir', './output')
                    datasets_dir = os.path.join(output_dir, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    filename = os.path.join(datasets_dir, f"validated_kenken_dataset_{difficulty}.json")
                    self.save_dataset_json(validated_puzzles, filename)
                    
                    # Save validation report
                    validation_report = {
                        'difficulty': difficulty,
                        'target_count': target_count,
                        'generated_count': len(validated_puzzles),
                        'timestamp': time.time(),
                        'validation_criteria': [
                            'latin_square_validation',
                            'cage_constraint_validation',
                            'strategy_requirement_matching',
                            'compositionality_verification',
                            'solvability_with_required_strategies',
                            'quality_assessment_score >= 0.7'
                        ]
                    }
                    
                    report_filename = os.path.join(datasets_dir, f"kenken_validation_report_{difficulty}.json")
                    with open(report_filename, 'w') as f:
                        json.dump(validation_report, f, indent=2)
                    
                    # Save images if configured
                    if self.should_save_images():
                        image_dir = os.path.join(output_dir, 'images', f"validated_mnist_kenken_{difficulty}")
                        self.save_images_with_metadata(validated_puzzles, image_dir)
                    
                    all_datasets[difficulty] = validated_puzzles
                    
                    print(f"\n✅ {difficulty.upper()} KEN KEN DATASET COMPLETE:")
                    print(f"  📁 Dataset: {filename}")
                    print(f"  📊 Report: {report_filename}")
                    print(f"  🎯 Puzzles: {len(validated_puzzles)}/{target_count}")
                    
                    # Show quality summary
                    self.show_kenken_quality_summary(validated_puzzles)
                
                else:
                    print(f"\n❌ Failed to generate any valid {difficulty} Ken Ken puzzles")
            
            # Overall summary
            total_time = time.time() - total_start_time
            total_validated = sum(len(dataset) for dataset in all_datasets.values())
            total_requested = sum(puzzle_counts.values())
            
            print(f"\n🏆 KEN KEN GENERATOR-ANALYZER COMBO COMPLETE!")
            print(f"  ⏱️ Total time: {total_time:.1f} seconds")
            print(f"  🎯 Total validated puzzles: {total_validated}/{total_requested}")
            print(f"  ✅ Success rate: {(total_validated/total_requested)*100:.1f}%")
            
            # Generate cross-dataset analysis
            if all_datasets:
                self.analyze_kenken_datasets(list(all_datasets.values()))
            
            return all_datasets
            
        except Exception as e:
            print(f"Error in Ken Ken generator-analyzer combo: {e}")
            traceback.print_exc()
            return {}
    
    def show_kenken_quality_summary(self, puzzles: List[Dict]):
        """Show quality summary for Ken Ken puzzles"""
        if not puzzles:
            return
        
        try:
            # Grid size distribution
            grid_sizes = [p['grid_size'] for p in puzzles]
            size_counts = {}
            for size in grid_sizes:
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"  📏 Grid sizes: {dict(size_counts)}")
            
            # Cage statistics
            cage_counts = [p['metadata']['num_cages'] for p in puzzles]
            avg_cages = np.mean(cage_counts)
            print(f"  🎯 Average cages per puzzle: {avg_cages:.1f}")
            
            # Operation distribution
            all_operations = []
            for puzzle in puzzles:
                all_operations.extend(puzzle['metadata']['operations_used'])
            
            from collections import Counter
            op_counts = Counter(all_operations)
            print(f"  🔢 Operations used: {dict(op_counts)}")
            
            # Difficulty scores
            difficulty_scores = [p['metadata']['difficulty_score'] for p in puzzles]
            avg_difficulty = np.mean(difficulty_scores)
            print(f"  ⭐ Average difficulty score: {avg_difficulty:.2f}")
            
        except Exception as e:
            print(f"  ⚠️ Error in quality summary: {e}")
    
    def analyze_kenken_datasets(self, datasets: List[List[Dict]]):
        """Analyze the validated Ken Ken datasets"""
        print(f"\n📊 KEN KEN VALIDATED DATASET ANALYSIS")
        print("-" * 40)
        
        try:
            all_puzzles = []
            for dataset in datasets:
                all_puzzles.extend(dataset)
            
            if not all_puzzles:
                return
            
            # Grid size analysis
            grid_sizes = [p['grid_size'] for p in all_puzzles]
            print(f"Grid sizes: {set(grid_sizes)} (range: {min(grid_sizes)}-{max(grid_sizes)})")
            
            # Strategy usage across all puzzles
            all_strategies = []
            for puzzle in all_puzzles:
                all_strategies.extend(puzzle['required_strategies'])
            
            from collections import Counter
            strategy_counts = Counter(all_strategies)
            print(f"\nMost used strategies:")
            for strategy, count in strategy_counts.most_common(5):
                print(f"  {strategy}: {count}")
            
            # Operation analysis
            all_operations = []
            for puzzle in all_puzzles:
                all_operations.extend(puzzle['metadata']['operations_used'])
            
            operation_counts = Counter(all_operations)
            print(f"\nOperations distribution:")
            for operation, count in operation_counts.most_common():
                print(f"  {operation}: {count}")
            
            # Compositionality verification
            print(f"\nCompositionality: {'✅ VERIFIED' if self.verify_kenken_compositionality() else '❌ FAILED'}")
            
        except Exception as e:
            print(f"Error in Ken Ken analysis: {e}")
    
    def verify_kenken_compositionality(self) -> bool:
        """Verify that the Ken Ken compositionality structure is correct"""
        try:
            # Check that moderate strategies compose easy strategies
            moderate_strategies = self.moderate_kb.get_all_strategies()
            easy_strategy_names = set(self.easy_kb.list_strategies())
            
            for name, info in moderate_strategies.items():
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    for component in composed_of:
                        if component not in easy_strategy_names and component not in moderate_strategies:
                            print(f"Ken Ken Compositionality error: {name} uses unknown component {component}")
                            return False
            
            # Check that hard strategies compose easy/moderate strategies
            hard_strategies = self.hard_kb.get_all_strategies()
            moderate_strategy_names = set(self.moderate_kb.list_strategies())
            
            for name, info in hard_strategies.items():
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    for component in composed_of:
                        if (component not in easy_strategy_names and 
                            component not in moderate_strategy_names and 
                            component not in hard_strategies):
                            print(f"Ken Ken Compositionality error: {name} uses unknown component {component}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying Ken Ken compositionality: {e}")
            return False
    
    def create_sample_puzzle(self, difficulty='easy', grid_size=4):
        """Create and display a sample Ken Ken puzzle"""
        print(f"\n" + "=" * 60)
        print(f"SAMPLE {difficulty.upper()} KEN KEN PUZZLE ({grid_size}x{grid_size})")
        print("=" * 60)
        
        try:
            dataset = self.generator.generate_guaranteed_valid_puzzles(difficulty, 1, grid_size)
            
            if not dataset:
                print("Failed to generate sample Ken Ken puzzle")
                return
            
            puzzle = dataset[0]
            puzzle_grid = np.array(puzzle['puzzle_grid'])
            solution_grid = np.array(puzzle['solution_grid'])
            cages = puzzle['cages']
            
            print(f"Puzzle ID: {puzzle['id']}")
            print(f"Grid Size: {puzzle['grid_size']}x{puzzle['grid_size']}")
            print(f"Strategies: {', '.join(puzzle['required_strategies'])}")
            print(f"Filled cells: {puzzle['metadata']['filled_cells']}")
            print(f"Number of cages: {puzzle['metadata']['num_cages']}")
            print(f"Operations: {', '.join(puzzle['metadata']['operations_used'])}")
            print()
            
            print("PUZZLE:")
            self.display_kenken_grid(puzzle_grid, cages)
            print()
            
            print("SOLUTION:")
            self.display_kenken_grid(solution_grid, cages)
            print()
            
            print("CAGES:")
            for i, cage in enumerate(cages, 1):
                cells_str = ', '.join(f"({r},{c})" for r, c in cage['cells'])
                print(f"  {i:2d}. {cage['operation']} = {cage['target']} | Cells: {cells_str}")
            
        except Exception as e:
            print(f"Error creating sample: {e}")
    
    def display_kenken_grid(self, grid, cages):
        """Display a Ken Ken grid with cage information"""
        grid_size = len(grid)
        
        # Create a mapping from cells to cage indices
        cell_to_cage = {}
        for i, cage in enumerate(cages):
            for r, c in cage['cells']:
                cell_to_cage[(r, c)] = i
        
        # Display the grid
        for i in range(grid_size):
            if i == 0:
                print("  " + "---+" * grid_size)
            
            row = ""
            for j in range(grid_size):
                if grid[i, j] == 0:
                    value = "."
                else:
                    value = str(grid[i, j])
                
                cage_idx = cell_to_cage.get((i, j), -1)
                cage_char = chr(ord('A') + cage_idx) if cage_idx >= 0 else '?'
                
                row += f"|{value}{cage_char}"
            row += "|"
            print("  " + row)
            print("  " + "---+" * grid_size)
    
    def test_solver(self, dataset_files, num_test=2):
        """Test the Ken Ken solver on generated puzzles"""
        print("\n" + "=" * 60)
        print("TESTING KEN KEN SOLVER")
        print("=" * 60)
        
        if not dataset_files:
            print("No datasets to test")
            return
        
        for dataset_file in dataset_files:
            if not os.path.exists(dataset_file):
                print(f"Dataset not found: {dataset_file}")
                continue
            
            try:
                with open(dataset_file, 'r') as f:
                    dataset = json.load(f)
                
                if not dataset:
                    continue
                
                difficulty = dataset[0]['difficulty']
                print(f"\nTesting {difficulty} Ken Ken puzzles...")
                
                test_puzzles = dataset[:min(num_test, len(dataset))]
                
                for i, puzzle_data in enumerate(test_puzzles):
                    print(f"  Puzzle {i+1}: {puzzle_data['id']}")
                    
                    puzzle_grid = np.array(puzzle_data['puzzle_grid'])
                    solution_grid = np.array(puzzle_data['solution_grid'])
                    cages = puzzle_data['cages']
                    required_strategies = puzzle_data['required_strategies']
                    
                    try:
                        # Test solver
                        solved_puzzle, used_strategies = self.solver.solve_puzzle(
                            puzzle_grid.copy(), 
                            cages,
                            required_strategies,
                            max_time_seconds=45
                        )
                        
                        if np.array_equal(solved_puzzle, solution_grid):
                            print(f"    ✅ SOLVED with strategies: {used_strategies}")
                        else:
                            empty_cells = np.sum(solved_puzzle == 0)
                            print(f"    ⚠️ PARTIAL: {empty_cells} cells remaining")
                            print(f"    Used strategies: {used_strategies}")
                    
                    except Exception as e:
                        print(f"    ❌ ERROR: {e}")
                
            except Exception as e:
                print(f"Error testing {dataset_file}: {e}")
    
    def run_full_pipeline(self):
        """Run the complete Ken Ken pipeline"""
        print("MNIST KEN KEN PROJECT - FULL PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Show knowledge bases
            self.show_knowledge_bases()
            
            # Step 2: Generate datasets
            generated_files = self.generate_valid_datasets()
            
            # Step 3: Test solver
            if generated_files:
                self.test_solver(generated_files, 2)
            
            # Step 4: Show samples
            for difficulty in ['easy', 'moderate', 'hard']:
                try:
                    grid_sizes = self.get_grid_sizes_for_difficulty(difficulty)
                    if difficulty == 'easy':
                        # Show both 4x4 and 5x5 for easy
                        for size in grid_sizes:
                            self.create_sample_puzzle(difficulty, size)
                    else:
                        # Single size for moderate and hard
                        self.create_sample_puzzle(difficulty, grid_sizes)
                except:
                    pass
            
            print(f"\nKEN KEN PIPELINE COMPLETE!")
            print(f"Generated files: {len(generated_files)}")
            for f in generated_files:
                if os.path.exists(f):
                    print(f"  - {f}")
            
        except Exception as e:
            print(f"Error in Ken Ken pipeline: {e}")
            traceback.print_exc()
    
    # Helper methods
    def get_puzzle_counts(self):
        """Get puzzle counts from configuration"""
        try:
            return self.config.get_puzzle_counts()
        except:
            return {'easy': 3, 'moderate': 2, 'hard': 1}
    
    def get_grid_sizes_for_difficulty(self, difficulty: str):
        """Get grid size(s) for difficulty - returns list for easy, int for others"""
        try:
            return self.config.get_grid_sizes(difficulty)
        except:
            if difficulty == 'easy':
                return [4, 5]
            else:
                defaults = {'moderate': 6, 'hard': 7}
                return defaults.get(difficulty, 6)
    
    def should_save_images(self):
        """Check if images should be saved"""
        try:
            output_settings = self.config.get_output_settings()
            return output_settings.get('images', {}).get('save_mnist_puzzles', True)
        except:
            return True
    
    def show_dataset_summary(self, dataset, difficulty):
        """Show summary of generated Ken Ken dataset"""
        if not dataset:
            return
        
        grid_sizes = [puzzle['grid_size'] for puzzle in dataset]
        filled_cells = [puzzle['metadata']['filled_cells'] for puzzle in dataset]
        num_cages = [puzzle['metadata']['num_cages'] for puzzle in dataset]
        
        print(f"  Grid sizes: {set(grid_sizes)}")
        print(f"  Average filled cells: {np.mean(filled_cells):.1f}")
        print(f"  Average cages: {np.mean(num_cages):.1f}")
        
        # Show strategy distribution
        all_strategies = []
        for puzzle in dataset:
            all_strategies.extend(puzzle['required_strategies'])
        
        from collections import Counter
        strategy_counts = Counter(all_strategies)
        print(f"  Most used strategies: {dict(strategy_counts.most_common(3))}")
        
        # Show operation distribution
        all_operations = []
        for puzzle in dataset:
            all_operations.extend(puzzle['metadata']['operations_used'])
        
        operation_counts = Counter(all_operations)
        print(f"  Operations: {dict(operation_counts)}")
    
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
    
    def save_dataset_json(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"💾 Ken Ken dataset saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
    
    def save_images_with_metadata(self, dataset: List[Dict], output_dir: str):
        """Save MNIST images with metadata for Ken Ken"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            metadata_dir = os.path.join(output_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)
            
            for entry in dataset:
                puzzle_id = entry['id']
                
                # Save images
                from PIL import Image
                puzzle_img = Image.fromarray(np.array(entry['mnist_puzzle'], dtype=np.uint8))
                solution_img = Image.fromarray(np.array(entry['mnist_solution'], dtype=np.uint8))
                
                puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
                solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
                
                puzzle_img.save(puzzle_path)
                solution_img.save(solution_path)
                
                # Create Ken Ken specific metadata
                metadata = {
                    'puzzle_info': {
                        'id': entry['id'],
                        'difficulty': entry['difficulty'],
                        'grid_size': entry['grid_size'],
                        'validation_status': 'VALID',
                        'generated_timestamp': entry['metadata']['generated_timestamp']
                    },
                    'grids': {
                        'puzzle_grid': entry['puzzle_grid'],
                        'solution_grid': entry['solution_grid']
                    },
                    'cages': entry['cages'],
                    'strategies': {
                        'required_strategies': entry['required_strategies'],
                        'strategy_details': entry['strategy_details']
                    },
                    'files': {
                        'puzzle_image': f"{puzzle_id}_puzzle.png",
                        'solution_image': f"{puzzle_id}_solution.png",
                        'puzzle_image_path': os.path.abspath(puzzle_path),
                        'solution_image_path': os.path.abspath(solution_path)
                    },
                    'statistics': {
                        'grid_size': entry['grid_size'],
                        'total_cells': entry['grid_size'] ** 2,
                        'filled_cells': entry['metadata']['filled_cells'],
                        'empty_cells': entry['metadata']['empty_cells'],
                        'fill_percentage': round((entry['metadata']['filled_cells'] / (entry['grid_size'] ** 2)) * 100, 1),
                        'num_cages': entry['metadata']['num_cages'],
                        'operations_used': entry['metadata']['operations_used'],
                        'difficulty_score': entry['metadata']['difficulty_score'],
                        'generation_attempt': entry['metadata']['generation_attempt']
                    }
                }
                
                metadata_path = os.path.join(metadata_dir, f"{puzzle_id}_metadata.json")
                # In save_images_with_metadata, find this line:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyEncoder)  # Add cls=NumpyEncoder
            
            print(f"🖼️ Ken Ken MNIST images and metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving Ken Ken images: {e}")

def save_dataset_json(dataset: List[Dict], filename: str):
    """Save dataset to JSON file with numpy compatibility"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2, cls=NumpyEncoder)
        
        print(f"💾 Ken Ken dataset saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MNIST Ken Ken Project")
    parser.add_argument('--config', default='kenken_config.yaml', help='Configuration file')
    parser.add_argument('--action', choices=[
        'show_kb', 'generate', 'analyze', 'test', 'sample', 'full', 'generate_validated'
    ], default='show_kb', help='Action to perform')
    parser.add_argument('--difficulty', choices=['easy', 'moderate', 'hard'], 
                       default='easy', help='Difficulty for sample')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for sample')
    
    args = parser.parse_args()
    
    try:
        # Initialize Ken Ken project
        project = MNISTKenKenProject(args.config)
        
        # Execute action
        if args.action == 'show_kb':
            project.show_knowledge_bases()
        
        elif args.action == 'generate':
            project.generate_valid_datasets()
        
        elif args.action == 'generate_validated':
            # NEW: Combined generator-analyzer action for Ken Ken
            project.run_generator_analyzer_combo()
        
        elif args.action == 'analyze':
            # Auto-detect Ken Ken dataset files
            output_dir = project.config.get('project.output_dir', './output')
            dataset_dir = os.path.join(output_dir, 'datasets')
            dataset_files = []
            
            for diff in ['easy', 'moderate', 'hard']:
                file_path = os.path.join(dataset_dir, f'kenken_dataset_{diff}.json')
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
            
            if dataset_files:
                print("Ken Ken analysis would be implemented here")
                print(f"Found datasets: {dataset_files}")
            else:
                print("No Ken Ken datasets found. Run --action generate first.")
        
        elif args.action == 'test':
            # Auto-detect Ken Ken dataset files
            output_dir = project.config.get('project.output_dir', './output')
            dataset_dir = os.path.join(output_dir, 'datasets')
            dataset_files = []
            
            for diff in ['easy', 'moderate', 'hard']:
                file_path = os.path.join(dataset_dir, f'kenken_dataset_{diff}.json')
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
            
            if dataset_files:
                project.test_solver(dataset_files)
            else:
                print("No Ken Ken datasets found. Run --action generate first.")
        
        elif args.action == 'sample':
            project.create_sample_puzzle(args.difficulty, args.grid_size)
        
        elif args.action == 'full':
            project.run_full_pipeline()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nKen Ken process interrupted by user")
        return 1
    except Exception as e:
        print(f"Ken Ken Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())