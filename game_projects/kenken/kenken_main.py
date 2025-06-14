# kenken_main.py
"""
MNIST KenKen Project - Main Orchestration Script
Main orchestration script with integrated validation for KenKen puzzles
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
    from game_projects.sudoku.config_manager import get_config, load_config
    from kenken_easy_strategies_kb import EasyKenKenStrategiesKB
    from kenken_moderate_strategies_kb import ModerateKenKenStrategiesKB
    from kenken_hard_strategies_kb import HardKenKenStrategiesKB
    from kenken_generator import MNISTKenKenGenerator
    # Make analyzer import optional for now
    try:
        from kenken_analyzer import KenKenDatasetAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError:
        print("Warning: KenKen analyzer not available, some features will be limited")
        ANALYZER_AVAILABLE = False
        KenKenDatasetAnalyzer = None
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all KenKen files are in the current directory")
    sys.exit(1)


class MNISTKenKenProject:
    """Main KenKen project class that orchestrates everything"""
    
    def __init__(self, config_file="config.yaml", grid_size=4):
        """Initialize the KenKen project"""
        print("=" * 60)
        print("INITIALIZING MNIST KENKEN PROJECT")
        print("=" * 60)
        
        self.grid_size = grid_size
        
        # Load configuration
        self.setup_configuration(config_file)
        
        # Initialize components
        self.setup_components()
        
        print("KenKen Project initialized successfully!")
    
    def setup_configuration(self, config_file):
        """Setup configuration system"""
        try:
            if not load_config(config_file):
                print("Creating default KenKen configuration...")
                self.create_default_config(config_file)
                if not load_config(config_file):
                    raise Exception("Could not load configuration")
            
            self.config = get_config()
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Configuration error: {e}")
            sys.exit(1)
    
    def create_default_config(self, config_file):
        """Create a default configuration file for KenKen"""
        default_config = """
project:
  name: "mnist-kenken"
  output_dir: "./output"
  data_dir: "./data"
  logs_dir: "./logs"

kenken:
  grid_size: 4
  puzzle_types: ["4x4", "5x5", "6x6"]

generation:
  num_puzzles:
    easy: 5
    moderate: 3
    hard: 2

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
            print(f"Created default KenKen config: {config_file}")
        except Exception as e:
            print(f"Could not create config: {e}")
            raise
    
    def setup_components(self):
        """Initialize all project components"""
        # Setup logging
        self.logger = self.config.setup_logging() if hasattr(self.config, 'setup_logging') else None
        
        # Create directories
        if hasattr(self.config, 'create_directories'):
            self.config.create_directories()
        
        # Initialize core components
        self.generator = MNISTKenKenGenerator(self.config, grid_size=self.grid_size)
        self.analyzer = KenKenDatasetAnalyzer()
        
        # Initialize knowledge bases
        self.easy_kb = EasyKenKenStrategiesKB()
        self.moderate_kb = ModerateKenKenStrategiesKB()
        self.hard_kb = HardKenKenStrategiesKB()
        
        sys.exit(1)
    
    def show_knowledge_bases(self):
        """Display all KenKen knowledge bases"""
        print("\n" + "=" * 60)
        print("KENKEN SOLVING STRATEGIES")
        print("=" * 60)
        
        try:
            # Easy strategies
            print("\n1. EASY KENKEN STRATEGIES")
            print("-" * 30)
            easy_strategies = self.easy_kb.get_all_strategies()
            for i, (name, info) in enumerate(easy_strategies.items(), 1):
                print(f"{i:2d}. {name}")
                print(f"    {info.get('description', 'N/A')}")
            
            # Moderate strategies  
            print("\n2. MODERATE KENKEN STRATEGIES")
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
            print("\n3. HARD KENKEN STRATEGIES")
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
            print(f"\nTotal KenKen strategies: {total}")
            print(f"Easy: {len(easy_strategies)}, Moderate: {len(moderate_strategies)}, Hard: {len(hard_strategies)}")
            
        except Exception as e:
            print(f"Error displaying knowledge bases: {e}")
    
    def generate_valid_datasets(self):
        """Generate KenKen datasets with guaranteed valid puzzles"""
        print("\n" + "=" * 60)
        print("GENERATING VALID KENKEN DATASETS")
        print("=" * 60)
        
        generated_files = []
        
        try:
            # Get puzzle counts from config
            puzzle_counts = self.get_puzzle_counts()
            
            total_start_time = time.time()
            
            for difficulty in ['easy', 'moderate', 'hard']:
                target_count = puzzle_counts.get(difficulty, 2)
                
                print(f"\nGenerating {target_count} VALID {difficulty} KenKen puzzles...")
                
                try:
                    # Generate guaranteed valid puzzles
                    dataset = self.generator.generate_guaranteed_valid_puzzles(difficulty, target_count)
                    
                    if not dataset:
                        print(f"Failed to generate {difficulty} KenKen puzzles")
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
                    
                    print(f"SUCCESS: Generated {len(dataset)}/{target_count} {difficulty} KenKen puzzles")
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
            
            print(f"\nKENKEN GENERATION COMPLETE!")
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
        """Show summary of generated KenKen dataset"""
        if not dataset:
            return
        
        filled_cells = [puzzle['metadata']['filled_cells'] for puzzle in dataset]
        strategy_counts = [len(puzzle['required_strategies']) for puzzle in dataset]
        cage_counts = [puzzle['metadata']['total_cages'] for puzzle in dataset]
        
        print(f"  Average filled cells: {np.mean(filled_cells):.1f}")
        print(f"  Average strategies: {np.mean(strategy_counts):.1f}")
        print(f"  Average cages: {np.mean(cage_counts):.1f}")
        
        # Show strategy distribution
        all_strategies = []
        for puzzle in dataset:
            all_strategies.extend(puzzle['required_strategies'])
        
        from collections import Counter
        strategy_counts = Counter(all_strategies)
        print(f"  Most used strategies: {dict(strategy_counts.most_common(3))}")
        
        # Show cage operations
        all_operations = []
        for puzzle in dataset:
            all_operations.extend(puzzle['metadata']['cage_operations'])
        
        operation_counts = Counter(all_operations)
        print(f"  Cage operations: {dict(operation_counts)}")
    
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
        """Analyze the generated KenKen datasets"""
        print("\n" + "=" * 60)
        print("ANALYZING KENKEN DATASETS")
        print("=" * 60)
        
        if not ANALYZER_AVAILABLE:
            print("❌ Analyzer not available - skipping analysis")
            return
        
        if not dataset_files:
            print("No datasets to analyze")
            return
        
        try:
            output_dir = self.config.get('project.output_dir', './output')
            reports_dir = os.path.join(output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate analysis report
            report_file = os.path.join(reports_dir, 'kenken_analysis_report.txt')
            self.analyzer.generate_report(dataset_files, report_file)
            print(f"KenKen analysis report saved to: {report_file}")
            
            # Generate visualizations
            for dataset_file in dataset_files:
                if os.path.exists(dataset_file):
                    try:
                        dataset = self.analyzer.load_dataset(dataset_file)
                        if dataset:
                            stats = self.analyzer.analyze_dataset_statistics(dataset)
                            difficulty = dataset[0]['difficulty']
                            
                            plot_dir = os.path.join(output_dir, 'analysis', f"kenken_plots_{difficulty}")
                            self.analyzer.generate_visualizations(stats, plot_dir)
                            print(f"KenKen plots saved to: {plot_dir}")
                    except Exception as e:
                        print(f"Error analyzing {dataset_file}: {e}")
            
            # Export strategy composition graph
            try:
                graph_file = os.path.join(output_dir, 'analysis', 'kenken_strategy_composition.json')
                os.makedirs(os.path.dirname(graph_file), exist_ok=True)
                self.analyzer.export_strategy_graph(graph_file)
                print(f"KenKen strategy graph saved to: {graph_file}")
            except Exception as e:
                print(f"Error creating strategy graph: {e}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
    
    def create_sample_puzzle(self, difficulty='easy'):
        """Create and display a sample KenKen puzzle"""
        print(f"\n" + "=" * 60)
        print(f"SAMPLE {difficulty.upper()} KENKEN PUZZLE")
        print("=" * 60)
        
        try:
            dataset = self.generator.generate_guaranteed_valid_puzzles(difficulty, 1)
            
            if not dataset:
                print("Failed to generate sample KenKen puzzle")
                return
            
            puzzle = dataset[0]
            puzzle_grid = np.array(puzzle['puzzle_grid'])
            solution_grid = np.array(puzzle['solution_grid'])
            cages = puzzle['cages']
            
            print(f"Puzzle ID: {puzzle['id']}")
            print(f"Grid Size: {puzzle['grid_size']}x{puzzle['grid_size']}")
            print(f"Strategies: {', '.join(puzzle['required_strategies'])}")
            print(f"Filled cells: {puzzle['metadata']['filled_cells']}")
            print(f"Total cages: {puzzle['metadata']['total_cages']}")
            print()
            
            print("PUZZLE:")
            self.display_kenken_grid(puzzle_grid, cages)
            print()
            
            print("SOLUTION:")
            self.display_grid(solution_grid)
            
            print("\nCAGE CONSTRAINTS:")
            for i, cage in enumerate(cages, 1):
                cells_str = ", ".join([f"({r+1},{c+1})" for r, c in cage['cells']])
                print(f"  Cage {i}: {cells_str} → {cage['operation']} = {cage['target']}")
            
        except Exception as e:
            print(f"Error creating sample: {e}")
    
    def display_kenken_grid(self, grid, cages):
        """Display a KenKen grid with cage information"""
        # Create cage mapping
        cage_map = {}
        for cage_id, cage in enumerate(cages):
            for r, c in cage['cells']:
                cage_map[(r, c)] = f"{cage['operation'][0].upper()}{cage['target']}"
        
        # Display grid
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if grid[i, j] == 0:
                    cell_val = "."
                else:
                    cell_val = str(grid[i, j])
                
                cage_info = cage_map.get((i, j), "")
                if cage_info:
                    cell_display = f"{cell_val}({cage_info})"
                else:
                    cell_display = f"{cell_val}     "
                
                row_str += f"{cell_display:>10}"
            print(row_str)
    
    def display_grid(self, grid):
        """Display a simple grid"""
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                if grid[i, j] == 0:
                    row += ". "
                else:
                    row += f"{grid[i, j]} "
            print(row)
    
    def run_full_pipeline(self):
        """Run the complete KenKen pipeline"""
        print("MNIST KENKEN PROJECT - FULL PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Show knowledge bases
            self.show_knowledge_bases()
            
            # Step 2: Generate datasets
            generated_files = self.generate_valid_datasets()
            
            # Step 3: Analyze datasets
            if generated_files:
                self.analyze_datasets(generated_files)
            
            # Step 4: Show samples
            for difficulty in ['easy', 'moderate', 'hard']:
                try:
                    self.create_sample_puzzle(difficulty)
                except:
                    pass
            
            print(f"\nKENKEN PIPELINE COMPLETE!")
            print(f"Generated files: {len(generated_files)}")
            for f in generated_files:
                if os.path.exists(f):
                    print(f"  - {f}")
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            traceback.print_exc()


def main():
    """Main function for KenKen"""
    parser = argparse.ArgumentParser(description="MNIST KenKen Project")
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--action', choices=[
        'show_kb', 'generate', 'analyze', 'sample', 'full'
    ], default='show_kb', help='Action to perform')
    parser.add_argument('--difficulty', choices=['easy', 'moderate', 'hard'], 
                       default='easy', help='Difficulty for sample')
    parser.add_argument('--grid-size', type=int, default=4, help='KenKen grid size')
    
    args = parser.parse_args()
    
    try:
        # Initialize project
        project = MNISTKenKenProject(args.config, grid_size=args.grid_size)
        
        # Execute action
        if args.action == 'show_kb':
            project.show_knowledge_bases()
        
        elif args.action == 'generate':
            project.generate_valid_datasets()
        
        elif args.action == 'analyze':
            # Auto-detect dataset files
            output_dir = project.config.get('project.output_dir', './output')
            dataset_dir = os.path.join(output_dir, 'datasets')
            dataset_files = []
            
            for diff in ['easy', 'moderate', 'hard']:
                file_path = os.path.join(dataset_dir, f'kenken_dataset_{diff}.json')
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
            
            if dataset_files:
                project.analyze_datasets(dataset_files)
            else:
                print("No KenKen datasets found. Run --action generate first.")
        
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