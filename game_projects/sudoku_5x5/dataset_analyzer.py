# dataset_analyzer.py
"""
Dataset Analyzer and Validator
Analyzes generated Sudoku datasets for compositionality and strategy usage
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd

from sudoku_easy_strategies_kb import EasyStrategiesKB
from sudoku_moderate_strategies_kb import ModerateStrategiesKB
from sudoku_hard_strategies_kb import HardStrategiesKB
from puzzle_solver import SudokuSolver

class DatasetAnalyzer:
    def __init__(self):
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.hard_kb = HardStrategiesKB()
        self.solver = SudokuSolver()
    
    def load_dataset(self, filename: str) -> List[Dict]:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def analyze_compositionality(self) -> Dict:
        """Analyze the compositionality relationships between strategies"""
        analysis = {
            'easy_strategies': {},
            'moderate_strategies': {},
            'hard_strategies': {},
            'composition_graph': {}
        }
        
        # Analyze easy strategies (base level)
        for strategy_name, strategy_info in self.easy_kb.get_all_strategies().items():
            analysis['easy_strategies'][strategy_name] = {
                'complexity': strategy_info['complexity'],
                'composite': strategy_info['composite'],
                'composed_of': strategy_info.get('composed_of', [])
            }
        
        # Analyze moderate strategies (composed of easy)
        for strategy_name, strategy_info in self.moderate_kb.get_all_strategies().items():
            analysis['moderate_strategies'][strategy_name] = {
                'complexity': strategy_info['complexity'],
                'composite': strategy_info['composite'],
                'composed_of': strategy_info.get('composed_of', [])
            }
            
            # Verify composition uses easy strategies
            if strategy_info['composite']:
                composed_of = strategy_info.get('composed_of', [])
                easy_strategies = set(self.easy_kb.list_strategies())
                moderate_strategies = set(self.moderate_kb.list_strategies())
                
                for component in composed_of:
                    if component not in easy_strategies and component not in moderate_strategies:
                        print(f"Warning: {strategy_name} uses unknown component: {component}")
        
        # Analyze hard strategies (composed of easy + moderate)
        for strategy_name, strategy_info in self.hard_kb.get_all_strategies().items():
            analysis['hard_strategies'][strategy_name] = {
                'complexity': strategy_info['complexity'],
                'composite': strategy_info['composite'],
                'composed_of': strategy_info.get('composed_of', [])
            }
            
            # Verify composition uses easy/moderate strategies
            if strategy_info['composite']:
                composed_of = strategy_info.get('composed_of', [])
                easy_strategies = set(self.easy_kb.list_strategies())
                moderate_strategies = set(self.moderate_kb.list_strategies())
                hard_strategies = set(self.hard_kb.list_strategies())
                
                for component in composed_of:
                    if component not in easy_strategies and component not in moderate_strategies and component not in hard_strategies:
                        print(f"Warning: {strategy_name} uses unknown component: {component}")
        
        return analysis
    
    def analyze_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Analyze statistics of the generated dataset"""
        stats = {
            'total_puzzles': len(dataset),
            'difficulty_distribution': Counter(),
            'strategy_usage': Counter(),
            'strategy_combinations': Counter(),
            'complexity_metrics': {},
            'validation_results': {}
        }
        
        # Basic statistics
        for puzzle in dataset:
            difficulty = puzzle['difficulty']
            strategies = puzzle['required_strategies']
            
            stats['difficulty_distribution'][difficulty] += 1
            
            for strategy in strategies:
                stats['strategy_usage'][strategy] += 1
            
            # Strategy combinations
            strategy_combo = tuple(sorted(strategies))
            stats['strategy_combinations'][strategy_combo] += 1
        
        # Complexity metrics
        stats['complexity_metrics'] = self.calculate_complexity_metrics(dataset)
        
        # Validation
        stats['validation_results'] = self.validate_dataset(dataset)
        
        return stats
    
    def calculate_complexity_metrics(self, dataset: List[Dict]) -> Dict:
        """Calculate complexity metrics for puzzles"""
        metrics = {
            'avg_strategies_per_puzzle': {},
            'avg_filled_cells': {},
            'strategy_depth_analysis': {}
        }
        
        by_difficulty = defaultdict(list)
        for puzzle in dataset:
            by_difficulty[puzzle['difficulty']].append(puzzle)
        
        for difficulty, puzzles in by_difficulty.items():
            # Average strategies per puzzle
            strategy_counts = [len(p['required_strategies']) for p in puzzles]
            metrics['avg_strategies_per_puzzle'][difficulty] = np.mean(strategy_counts)
            
            # Average filled cells
            filled_counts = []
            for puzzle in puzzles:
                grid = np.array(puzzle['puzzle_grid'])
                filled_counts.append(np.sum(grid != 0))
            metrics['avg_filled_cells'][difficulty] = np.mean(filled_counts)
            
            # Strategy depth analysis
            metrics['strategy_depth_analysis'][difficulty] = self.analyze_strategy_depth(puzzles)
        
        return metrics
    
    def analyze_strategy_depth(self, puzzles: List[Dict]) -> Dict:
        """Analyze the depth of strategy composition"""
        depth_analysis = {
            'max_depth': 0,
            'avg_depth': 0,
            'depth_distribution': Counter()
        }
        
        depths = []
        for puzzle in puzzles:
            puzzle_max_depth = 0
            for strategy in puzzle['required_strategies']:
                depth = self.get_strategy_depth(strategy)
                puzzle_max_depth = max(puzzle_max_depth, depth)
                depth_analysis['depth_distribution'][depth] += 1
            depths.append(puzzle_max_depth)
        
        depth_analysis['max_depth'] = max(depths) if depths else 0
        depth_analysis['avg_depth'] = np.mean(depths) if depths else 0
        
        return depth_analysis
    
    def get_strategy_depth(self, strategy_name: str) -> int:
        """Get the composition depth of a strategy"""
        # Check easy strategies (depth 0)
        if strategy_name in self.easy_kb.list_strategies():
            strategy_info = self.easy_kb.get_strategy(strategy_name)
            if not strategy_info.get('composite', False):
                return 0
        
        # Check moderate strategies
        if strategy_name in self.moderate_kb.list_strategies():
            strategy_info = self.moderate_kb.get_strategy(strategy_name)
            if strategy_info.get('composite', False):
                composed_of = strategy_info.get('composed_of', [])
                if composed_of:
                    max_component_depth = max(self.get_strategy_depth(comp) for comp in composed_of)
                    return max_component_depth + 1
                return 1
        
        # Check hard strategies
        if strategy_name in self.hard_kb.list_strategies():
            strategy_info = self.hard_kb.get_strategy(strategy_name)
            if strategy_info.get('composite', False):
                composed_of = strategy_info.get('composed_of', [])
                if composed_of:
                    max_component_depth = max(self.get_strategy_depth(comp) for comp in composed_of)
                    return max_component_depth + 1
                return 2  # Default depth for hard strategies
        
        return 0  # Unknown strategy
    
    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """Validate puzzles in the dataset"""
        validation = {
            'valid_puzzles': 0,
            'invalid_puzzles': 0,
            'solvability_check': {},
            'solution_correctness': {},
            'errors': []
        }
        
        for i, puzzle in enumerate(dataset):
            try:
                # Check puzzle format
                puzzle_grid = np.array(puzzle['puzzle_grid'])
                solution_grid = np.array(puzzle['solution_grid'])
                
                if puzzle_grid.shape != (9, 9) or solution_grid.shape != (9, 9):
                    validation['errors'].append(f"Puzzle {i}: Invalid grid dimensions")
                    validation['invalid_puzzles'] += 1
                    continue
                
                # Check solution correctness
                if self.solver.validate_solution(solution_grid):
                    validation['solution_correctness'][f"puzzle_{i}"] = True
                else:
                    validation['solution_correctness'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Invalid solution")
                
                # Check if puzzle is solvable with required strategies
                solved_puzzle, used_strategies = self.solver.solve_puzzle(
                    puzzle_grid.copy(), 
                    puzzle['required_strategies']
                )
                
                if np.array_equal(solved_puzzle, solution_grid):
                    validation['solvability_check'][f"puzzle_{i}"] = True
                    validation['valid_puzzles'] += 1
                else:
                    validation['solvability_check'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Cannot be solved with required strategies")
                    validation['invalid_puzzles'] += 1
                
            except Exception as e:
                validation['errors'].append(f"Puzzle {i}: Error during validation - {str(e)}")
                validation['invalid_puzzles'] += 1
        
        return validation
    
    def generate_visualizations(self, stats: Dict, output_dir: str = "analysis_plots"):
        """Generate visualization plots for the analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy usage distribution
        plt.figure(figsize=(12, 8))
        strategies = list(stats['strategy_usage'].keys())
        counts = list(stats['strategy_usage'].values())
        
        plt.barh(strategies, counts)
        plt.title('Strategy Usage Distribution')
        plt.xlabel('Number of Puzzles')
        plt.ylabel('Strategy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/strategy_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Difficulty distribution
        plt.figure(figsize=(8, 6))
        difficulties = list(stats['difficulty_distribution'].keys())
        counts = list(stats['difficulty_distribution'].values())
        
        plt.pie(counts, labels=difficulties, autopct='%1.1f%%')
        plt.title('Difficulty Distribution')
        plt.savefig(f"{output_dir}/difficulty_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Complexity metrics
        if 'complexity_metrics' in stats:
            metrics = stats['complexity_metrics']
            
            # Strategies per puzzle by difficulty
            plt.figure(figsize=(10, 6))
            difficulties = list(metrics['avg_strategies_per_puzzle'].keys())
            avg_strategies = list(metrics['avg_strategies_per_puzzle'].values())
            
            plt.bar(difficulties, avg_strategies)
            plt.title('Average Strategies per Puzzle by Difficulty')
            plt.xlabel('Difficulty')
            plt.ylabel('Average Number of Strategies')
            plt.savefig(f"{output_dir}/avg_strategies_by_difficulty.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, dataset_files: List[str], output_file: str = "analysis_report.txt"):
        """Generate a comprehensive analysis report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MNIST SUDOKU DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Analyze compositionality
        compositionality_analysis = self.analyze_compositionality()
        report_lines.append("1. COMPOSITIONALITY ANALYSIS")
        report_lines.append("-" * 40)
        
        report_lines.append(f"Easy Strategies (Base Level): {len(compositionality_analysis['easy_strategies'])}")
        for name, info in compositionality_analysis['easy_strategies'].items():
            report_lines.append(f"  - {name}: {'Composite' if info['composite'] else 'Atomic'}")
        
        report_lines.append(f"\nModerate Strategies (Level 1): {len(compositionality_analysis['moderate_strategies'])}")
        for name, info in compositionality_analysis['moderate_strategies'].items():
            if info['composite']:
                components = ', '.join(info['composed_of'])
                report_lines.append(f"  - {name}: Composed of [{components}]")
            else:
                report_lines.append(f"  - {name}: Atomic")
        
        report_lines.append(f"\nHard Strategies (Level 2+): {len(compositionality_analysis['hard_strategies'])}")
        for name, info in compositionality_analysis['hard_strategies'].items():
            if info['composite']:
                components = ', '.join(info['composed_of'])
                report_lines.append(f"  - {name}: Composed of [{components}]")
            else:
                report_lines.append(f"  - {name}: Atomic")
        
        report_lines.append("")
        
        # Analyze each dataset
        all_stats = {}
        for dataset_file in dataset_files:
            try:
                dataset = self.load_dataset(dataset_file)
                stats = self.analyze_dataset_statistics(dataset)
                all_stats[dataset_file] = stats
                
                difficulty = dataset[0]['difficulty'] if dataset else 'unknown'
                
                report_lines.append(f"2. DATASET ANALYSIS: {dataset_file.upper()}")
                report_lines.append("-" * 40)
                report_lines.append(f"Difficulty Level: {difficulty}")
                report_lines.append(f"Total Puzzles: {stats['total_puzzles']}")
                report_lines.append("")
                
                # Strategy usage
                report_lines.append("Strategy Usage:")
                for strategy, count in stats['strategy_usage'].most_common():
                    percentage = (count / stats['total_puzzles']) * 100
                    report_lines.append(f"  {strategy}: {count} ({percentage:.1f}%)")
                report_lines.append("")
                
                # Complexity metrics
                if 'complexity_metrics' in stats:
                    metrics = stats['complexity_metrics']
                    report_lines.append("Complexity Metrics:")
                    for diff, avg_strategies in metrics['avg_strategies_per_puzzle'].items():
                        report_lines.append(f"  Average strategies per {diff} puzzle: {avg_strategies:.2f}")
                    for diff, avg_filled in metrics['avg_filled_cells'].items():
                        report_lines.append(f"  Average filled cells in {diff} puzzles: {avg_filled:.1f}")
                    report_lines.append("")
                
                # Validation results
                if 'validation_results' in stats:
                    validation = stats['validation_results']
                    valid_rate = (validation['valid_puzzles'] / stats['total_puzzles']) * 100
                    report_lines.append(f"Validation Results:")
                    report_lines.append(f"  Valid puzzles: {validation['valid_puzzles']} ({valid_rate:.1f}%)")
                    report_lines.append(f"  Invalid puzzles: {validation['invalid_puzzles']}")
                    
                    if validation['errors']:
                        report_lines.append("  Errors found:")
                        for error in validation['errors'][:5]:  # Show first 5 errors
                            report_lines.append(f"    - {error}")
                        if len(validation['errors']) > 5:
                            report_lines.append(f"    ... and {len(validation['errors']) - 5} more errors")
                    report_lines.append("")
                
                report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"Error analyzing {dataset_file}: {str(e)}")
                report_lines.append("")
        
        # Cross-dataset comparison
        if len(all_stats) > 1:
            report_lines.append("3. CROSS-DATASET COMPARISON")
            report_lines.append("-" * 40)
            
            # Compare strategy diversity
            all_strategies = set()
            for stats in all_stats.values():
                all_strategies.update(stats['strategy_usage'].keys())
            
            report_lines.append(f"Total unique strategies across all datasets: {len(all_strategies)}")
            
            # Strategy progression analysis
            easy_strategies = set()
            moderate_strategies = set()
            hard_strategies = set()
            
            for filename, stats in all_stats.items():
                if 'easy' in filename.lower():
                    easy_strategies.update(stats['strategy_usage'].keys())
                elif 'moderate' in filename.lower():
                    moderate_strategies.update(stats['strategy_usage'].keys())
                elif 'hard' in filename.lower():
                    hard_strategies.update(stats['strategy_usage'].keys())
            
            report_lines.append(f"Easy dataset strategies: {len(easy_strategies)}")
            report_lines.append(f"Moderate dataset strategies: {len(moderate_strategies)}")
            report_lines.append(f"Hard dataset strategies: {len(hard_strategies)}")
            
            # Check compositionality
            moderate_uses_easy = moderate_strategies.intersection(easy_strategies)
            hard_uses_easy = hard_strategies.intersection(easy_strategies)
            hard_uses_moderate = hard_strategies.intersection(moderate_strategies)
            
            report_lines.append(f"Moderate strategies reusing easy: {len(moderate_uses_easy)}")
            report_lines.append(f"Hard strategies reusing easy: {len(hard_uses_easy)}")
            report_lines.append(f"Hard strategies reusing moderate: {len(hard_uses_moderate)}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("4. RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Check for potential issues
        issues_found = []
        recommendations = []
        
        for filename, stats in all_stats.items():
            validation = stats.get('validation_results', {})
            if validation.get('invalid_puzzles', 0) > 0:
                invalid_rate = (validation['invalid_puzzles'] / stats['total_puzzles']) * 100
                if invalid_rate > 10:
                    issues_found.append(f"High invalid puzzle rate in {filename}: {invalid_rate:.1f}%")
                    recommendations.append(f"Review puzzle generation logic for {filename}")
            
            # Check strategy distribution
            strategy_counts = list(stats['strategy_usage'].values())
            if len(strategy_counts) > 0:
                min_usage = min(strategy_counts)
                max_usage = max(strategy_counts)
                if max_usage > min_usage * 5:  # High imbalance
                    issues_found.append(f"Unbalanced strategy usage in {filename}")
                    recommendations.append(f"Balance strategy selection in {filename}")
        
        if issues_found:
            report_lines.append("Issues Found:")
            for issue in issues_found:
                report_lines.append(f"  - {issue}")
            report_lines.append("")
        
        if recommendations:
            report_lines.append("Recommendations:")
            for rec in recommendations:
                report_lines.append(f"  - {rec}")
        else:
            report_lines.append("No significant issues found. Dataset appears well-balanced.")
        
        report_lines.append("")
        report_lines.append("5. DATASET SUMMARY")
        report_lines.append("-" * 40)
        
        total_puzzles = sum(stats['total_puzzles'] for stats in all_stats.values())
        total_strategies = len(set().union(*[stats['strategy_usage'].keys() for stats in all_stats.values()]))
        
        report_lines.append(f"Total puzzles across all datasets: {total_puzzles}")
        report_lines.append(f"Total unique strategies used: {total_strategies}")
        report_lines.append(f"Datasets analyzed: {len(all_stats)}")
        
        # Compositionality verification
        composition_verified = self.verify_compositionality()
        report_lines.append(f"Compositionality structure verified: {'Yes' if composition_verified else 'No'}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Analysis report saved to {output_file}")
        return '\n'.join(report_lines)
    
    def verify_compositionality(self) -> bool:
        """Verify that the compositionality structure is correct"""
        try:
            # Check that moderate strategies compose easy strategies
            moderate_strategies = self.moderate_kb.get_all_strategies()
            easy_strategy_names = set(self.easy_kb.list_strategies())
            
            for name, info in moderate_strategies.items():
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    for component in composed_of:
                        if component not in easy_strategy_names and component not in moderate_strategies:
                            print(f"Compositionality error: {name} uses unknown component {component}")
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
                            print(f"Compositionality error: {name} uses unknown component {component}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying compositionality: {e}")
            return False
    
    def export_strategy_graph(self, output_file: str = "strategy_composition_graph.json"):
        """Export strategy composition as a graph structure"""
        graph = {
            'nodes': [],
            'edges': [],
            'levels': {
                'easy': [],
                'moderate': [],
                'hard': []
            }
        }
        
        # Add easy strategy nodes
        for name, info in self.easy_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'easy',
                'composite': info.get('composite', False),
                'description': info.get('description', '')
            })
            graph['levels']['easy'].append(name)
        
        # Add moderate strategy nodes and edges
        for name, info in self.moderate_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'moderate',
                'composite': info.get('composite', False),
                'description': info.get('description', '')
            })
            graph['levels']['moderate'].append(name)
            
            # Add composition edges
            if info.get('composite', False):
                for component in info.get('composed_of', []):
                    graph['edges'].append({
                        'source': component,
                        'target': name,
                        'type': 'composition'
                    })
        
        # Add hard strategy nodes and edges
        for name, info in self.hard_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'hard',
                'composite': info.get('composite', False),
                'description': info.get('description', '')
            })
            graph['levels']['hard'].append(name)
            
            # Add composition edges
            if info.get('composite', False):
                for component in info.get('composed_of', []):
                    graph['edges'].append({
                        'source': component,
                        'target': name,
                        'type': 'composition'
                    })
        
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        print(f"Strategy composition graph exported to {output_file}")
        return graph

def main():
    """Main function for dataset analysis"""
    analyzer = DatasetAnalyzer()
    
    # Define dataset files to analyze
    dataset_files = [
        "sudoku_dataset_easy.json",
        "sudoku_dataset_moderate.json",
        "sudoku_dataset_hard.json"
    ]
    
    print("Starting dataset analysis...")
    
    # Generate comprehensive report
    report = analyzer.generate_report(dataset_files)
    
    # Generate visualizations for each dataset
    for dataset_file in dataset_files:
        try:
            dataset = analyzer.load_dataset(dataset_file)
            stats = analyzer.analyze_dataset_statistics(dataset)
            
            # Create output directory for this dataset
            difficulty = dataset[0]['difficulty'] if dataset else 'unknown'
            output_dir = f"analysis_plots_{difficulty}"
            
            analyzer.generate_visualizations(stats, output_dir)
            print(f"Visualizations saved to {output_dir}/")
            
        except FileNotFoundError:
            print(f"Dataset file {dataset_file} not found, skipping...")
        except Exception as e:
            print(f"Error analyzing {dataset_file}: {e}")
    
    # Export strategy composition graph
    analyzer.export_strategy_graph()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- analysis_report.txt")
    print("- strategy_composition_graph.json")
    print("- analysis_plots_[difficulty]/ directories")
    print("\nCompositionality verification:", 
          "PASSED" if analyzer.verify_compositionality() else "FAILED")

if __name__ == "__main__":
    main()