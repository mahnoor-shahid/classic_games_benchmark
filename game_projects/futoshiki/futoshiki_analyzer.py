# futoshiki_analyzer.py
"""
Dataset Analyzer and Validator for Futoshiki
Analyzes generated Futoshiki datasets for compositionality and strategy usage
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd

from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from futoshiki_hard_strategies_kb import FutoshikiHardStrategiesKB
from futoshiki_solver import FutoshikiSolver

class FutoshikiDatasetAnalyzer:
    def __init__(self):
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        self.solver = FutoshikiSolver()
    
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
                'composed_of': strategy_info.get('composed_of', []),
                'prerequisites': strategy_info.get('prerequisites', []),
                'applies_to': strategy_info.get('applies_to', [])
            }
        
        # Analyze moderate strategies (composed of easy)
        for strategy_name, strategy_info in self.moderate_kb.get_all_strategies().items():
            analysis['moderate_strategies'][strategy_name] = {
                'complexity': strategy_info['complexity'],
                'composite': strategy_info['composite'],
                'composed_of': strategy_info.get('composed_of', []),
                'prerequisites': strategy_info.get('prerequisites', []),
                'applies_to': strategy_info.get('applies_to', []),
                'constraint_aware': strategy_info.get('constraint_aware', False)
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
                'composed_of': strategy_info.get('composed_of', []),
                'prerequisites': strategy_info.get('prerequisites', []),
                'applies_to': strategy_info.get('applies_to', []),
                'constraint_aware': strategy_info.get('constraint_aware', False)
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
            'constraint_analysis': {},
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
        
        # Constraint analysis
        stats['constraint_analysis'] = self.analyze_constraints(dataset)
        
        # Validation
        stats['validation_results'] = self.validate_dataset(dataset)
        
        return stats
    
    def calculate_complexity_metrics(self, dataset: List[Dict]) -> Dict:
        """Calculate complexity metrics for puzzles"""
        metrics = {
            'avg_strategies_per_puzzle': {},
            'avg_filled_cells': {},
            'avg_constraints': {},
            'fill_ratio_distribution': {},
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
            filled_counts = [p['metadata']['filled_cells'] for p in puzzles]
            metrics['avg_filled_cells'][difficulty] = np.mean(filled_counts)
            
            # Average constraints
            constraint_counts = [p['metadata']['total_constraints'] for p in puzzles]
            metrics['avg_constraints'][difficulty] = np.mean(constraint_counts)
            
            # Fill ratio distribution
            fill_ratios = [p['metadata']['fill_ratio'] for p in puzzles]
            metrics['fill_ratio_distribution'][difficulty] = {
                'mean': np.mean(fill_ratios),
                'std': np.std(fill_ratios),
                'min': np.min(fill_ratios),
                'max': np.max(fill_ratios)
            }
            
            # Strategy depth analysis
            metrics['strategy_depth_analysis'][difficulty] = self.analyze_strategy_depth(puzzles)
        
        return metrics
    
    def analyze_constraints(self, dataset: List[Dict]) -> Dict:
        """Analyze constraint patterns in the dataset"""
        constraint_analysis = {
            'horizontal_constraint_distribution': Counter(),
            'vertical_constraint_distribution': Counter(),
            'constraint_density_by_difficulty': {},
            'constraint_type_preferences': {}
        }
        
        by_difficulty = defaultdict(list)
        for puzzle in dataset:
            by_difficulty[puzzle['difficulty']].append(puzzle)
        
        for difficulty, puzzles in by_difficulty.items():
            h_constraint_counts = []
            v_constraint_counts = []
            total_constraint_counts = []
            constraint_types = Counter()
            
            for puzzle in puzzles:
                # Count constraints
                h_count = puzzle['metadata']['num_h_constraints']
                v_count = puzzle['metadata']['num_v_constraints']
                total_count = puzzle['metadata']['total_constraints']
                
                h_constraint_counts.append(h_count)
                v_constraint_counts.append(v_count)
                total_constraint_counts.append(total_count)
                
                # Analyze constraint types
                for constraint in puzzle['h_constraints'].values():
                    constraint_types[f"horizontal_{constraint}"] += 1
                for constraint in puzzle['v_constraints'].values():
                    constraint_types[f"vertical_{constraint}"] += 1
            
            constraint_analysis['constraint_density_by_difficulty'][difficulty] = {
                'avg_horizontal': np.mean(h_constraint_counts),
                'avg_vertical': np.mean(v_constraint_counts),
                'avg_total': np.mean(total_constraint_counts),
                'std_total': np.std(total_constraint_counts)
            }
            
            constraint_analysis['constraint_type_preferences'][difficulty] = dict(constraint_types)
        
        return constraint_analysis
    
    def analyze_strategy_depth(self, puzzles: List[Dict]) -> Dict:
        """Analyze the depth of strategy composition"""
        depth_analysis = {
            'max_depth': 0,
            'avg_depth': 0,
            'depth_distribution': Counter(),
            'constraint_aware_strategies': 0
        }
        
        depths = []
        constraint_aware_count = 0
        
        for puzzle in puzzles:
            puzzle_max_depth = 0
            for strategy in puzzle['required_strategies']:
                depth = self.get_strategy_depth(strategy)
                puzzle_max_depth = max(puzzle_max_depth, depth)
                depth_analysis['depth_distribution'][depth] += 1
                
                # Check if strategy is constraint-aware
                if self.is_constraint_aware_strategy(strategy):
                    constraint_aware_count += 1
            
            depths.append(puzzle_max_depth)
        
        depth_analysis['max_depth'] = max(depths) if depths else 0
        depth_analysis['avg_depth'] = np.mean(depths) if depths else 0
        depth_analysis['constraint_aware_strategies'] = constraint_aware_count
        
        return depth_analysis
    
    def get_strategy_depth(self, strategy_name: str) -> int:
        """Get the composition depth of a strategy"""
        # Check easy strategies (depth 0)
        if strategy_name in self.easy_kb.list_strategies():
            strategy_info = self.easy_kb.get_strategy(strategy_name)
            if not strategy_info.get('composite', False):
                return 0
            else:
                return 1  # Composite easy strategies are depth 1
        
        # Check moderate strategies (depth 1-2)
        if strategy_name in self.moderate_kb.list_strategies():
            strategy_info = self.moderate_kb.get_strategy(strategy_name)
            if strategy_info.get('composite', False):
                composed_of = strategy_info.get('composed_of', [])
                if composed_of:
                    max_component_depth = max(self.get_strategy_depth(comp) for comp in composed_of)
                    return max_component_depth + 1
                return 1
            return 1
        
        # Check hard strategies (depth 2+)
        if strategy_name in self.hard_kb.list_strategies():
            strategy_info = self.hard_kb.get_strategy(strategy_name)
            if strategy_info.get('composite', False):
                composed_of = strategy_info.get('composed_of', [])
                if composed_of:
                    max_component_depth = max(self.get_strategy_depth(comp) for comp in composed_of)
                    return max_component_depth + 1
                return 2  # Default depth for hard strategies
            return 2
        
        return 0  # Unknown strategy
    
    def is_constraint_aware_strategy(self, strategy_name: str) -> bool:
        """Check if a strategy is constraint-aware"""
        # Check moderate strategies
        if strategy_name in self.moderate_kb.list_strategies():
            strategy_info = self.moderate_kb.get_strategy(strategy_name)
            return strategy_info.get('constraint_aware', False)
        
        # Check hard strategies
        if strategy_name in self.hard_kb.list_strategies():
            strategy_info = self.hard_kb.get_strategy(strategy_name)
            return strategy_info.get('constraint_aware', False)
        
        # Easy strategies that involve constraints
        constraint_aware_easy = {
            'constraint_propagation', 'forced_by_inequality', 
            'minimum_maximum_bounds', 'direct_constraint_forcing'
        }
        
        return strategy_name in constraint_aware_easy
    
    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """Validate puzzles in the dataset"""
        validation = {
            'valid_puzzles': 0,
            'invalid_puzzles': 0,
            'solvability_check': {},
            'solution_correctness': {},
            'constraint_consistency': {},
            'errors': []
        }
        
        for i, puzzle in enumerate(dataset):
            try:
                # Check puzzle format
                puzzle_grid = np.array(puzzle['puzzle_grid'])
                solution_grid = np.array(puzzle['solution_grid'])
                
                if puzzle_grid.shape != (puzzle['size'], puzzle['size']) or solution_grid.shape != (puzzle['size'], puzzle['size']):
                    validation['errors'].append(f"Puzzle {i}: Invalid grid dimensions")
                    validation['invalid_puzzles'] += 1
                    continue
                
                # Parse constraints
                h_constraints = {}
                v_constraints = {}
                
                for key, value in puzzle['h_constraints'].items():
                    row, col = map(int, key.split(','))
                    h_constraints[(row, col)] = value
                
                for key, value in puzzle['v_constraints'].items():
                    row, col = map(int, key.split(','))
                    v_constraints[(row, col)] = value
                
                # Check solution correctness
                if self.solver.validate_solution(solution_grid, h_constraints, v_constraints):
                    validation['solution_correctness'][f"puzzle_{i}"] = True
                else:
                    validation['solution_correctness'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Invalid solution")
                
                # Check constraint consistency
                consistent = self.check_constraint_consistency(puzzle_grid, h_constraints, v_constraints)
                validation['constraint_consistency'][f"puzzle_{i}"] = consistent
                
                if not consistent:
                    validation['errors'].append(f"Puzzle {i}: Constraint inconsistency")
                
                # Check if puzzle is solvable with required strategies
                try:
                    solved_puzzle, used_strategies = self.solver.solve_puzzle(
                        puzzle_grid.copy(), h_constraints, v_constraints,
                        puzzle['required_strategies'], max_time_seconds=5
                    )
                    
                    if np.array_equal(solved_puzzle, solution_grid):
                        validation['solvability_check'][f"puzzle_{i}"] = True
                        validation['valid_puzzles'] += 1
                    else:
                        validation['solvability_check'][f"puzzle_{i}"] = False
                        validation['errors'].append(f"Puzzle {i}: Cannot be solved with required strategies")
                        validation['invalid_puzzles'] += 1
                except:
                    validation['solvability_check'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Solver error")
                    validation['invalid_puzzles'] += 1
                
            except Exception as e:
                validation['errors'].append(f"Puzzle {i}: Error during validation - {str(e)}")
                validation['invalid_puzzles'] += 1
        
        return validation
    
    def check_constraint_consistency(self, grid: np.ndarray, h_constraints: Dict, v_constraints: Dict) -> bool:
        """Check if filled cells are consistent with constraints"""
        try:
            # Check horizontal constraints
            for (row, col), constraint in h_constraints.items():
                if col + 1 < len(grid):
                    left_val = grid[row, col]
                    right_val = grid[row, col + 1]
                    
                    # Only check if both cells are filled
                    if left_val != 0 and right_val != 0:
                        if constraint == '<' and left_val >= right_val:
                            return False
                        elif constraint == '>' and left_val <= right_val:
                            return False
            
            # Check vertical constraints
            for (row, col), constraint in v_constraints.items():
                if row + 1 < len(grid):
                    top_val = grid[row, col]
                    bottom_val = grid[row + 1, col]
                    
                    # Only check if both cells are filled
                    if top_val != 0 and bottom_val != 0:
                        if constraint == '<' and top_val >= bottom_val:
                            return False
                        elif constraint == '>' and top_val <= bottom_val:
                            return False
            
            return True
        except:
            return False
    
    def generate_visualizations(self, stats: Dict, output_dir: str = "futoshiki_analysis_plots"):
        """Generate visualization plots for the analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy usage distribution
        plt.figure(figsize=(14, 8))
        strategies = list(stats['strategy_usage'].keys())
        counts = list(stats['strategy_usage'].values())
        
        plt.barh(strategies, counts)
        plt.title('Futoshiki Strategy Usage Distribution')
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
        plt.title('Futoshiki Difficulty Distribution')
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
            
            # Fill ratio distribution
            plt.figure(figsize=(12, 8))
            for i, (difficulty, fill_data) in enumerate(metrics['fill_ratio_distribution'].items()):
                plt.subplot(2, 2, i+1)
                # Create sample distribution for visualization
                np.random.seed(42)
                sample_data = np.random.normal(fill_data['mean'], fill_data['std'], 100)
                sample_data = np.clip(sample_data, fill_data['min'], fill_data['max'])
                
                plt.hist(sample_data, bins=20, alpha=0.7, label=difficulty)
                plt.title(f'{difficulty.capitalize()} Fill Ratio Distribution')
                plt.xlabel('Fill Ratio')
                plt.ylabel('Frequency')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/fill_ratio_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Constraint analysis
        if 'constraint_analysis' in stats:
            constraint_stats = stats['constraint_analysis']
            
            # Constraint density by difficulty
            plt.figure(figsize=(10, 6))
            difficulties = list(constraint_stats['constraint_density_by_difficulty'].keys())
            horizontal_counts = [constraint_stats['constraint_density_by_difficulty'][d]['avg_horizontal'] for d in difficulties]
            vertical_counts = [constraint_stats['constraint_density_by_difficulty'][d]['avg_vertical'] for d in difficulties]
            
            x = np.arange(len(difficulties))
            width = 0.35
            
            plt.bar(x - width/2, horizontal_counts, width, label='Horizontal')
            plt.bar(x + width/2, vertical_counts, width, label='Vertical')
            
            plt.title('Average Constraints per Puzzle by Difficulty')
            plt.xlabel('Difficulty')
            plt.ylabel('Average Number of Constraints')
            plt.xticks(x, difficulties)
            plt.legend()
            plt.savefig(f"{output_dir}/constraints_by_difficulty.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, dataset_files: List[str], output_file: str = "futoshiki_analysis_report.txt"):
        """Generate a comprehensive analysis report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MNIST FUTOSHIKI DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Analyze compositionality
        compositionality_analysis = self.analyze_compositionality()
        report_lines.append("1. COMPOSITIONALITY ANALYSIS")
        report_lines.append("-" * 40)
        
        report_lines.append(f"Easy Strategies (Base Level): {len(compositionality_analysis['easy_strategies'])}")
        for name, info in compositionality_analysis['easy_strategies'].items():
            if info['composite']:
                components = ', '.join(info['composed_of']) if info['composed_of'] else 'None'
                report_lines.append(f"  - {name}: Composed of [{components}]")
            else:
                report_lines.append(f"  - {name}: Atomic")
        
        report_lines.append(f"\nModerate Strategies (Level 1): {len(compositionality_analysis['moderate_strategies'])}")
        for name, info in compositionality_analysis['moderate_strategies'].items():
            constraint_aware = "Constraint-aware" if info.get('constraint_aware', False) else "Basic"
            if info['composite']:
                components = ', '.join(info['composed_of']) if info['composed_of'] else 'None'
                report_lines.append(f"  - {name}: Composed of [{components}] ({constraint_aware})")
            else:
                report_lines.append(f"  - {name}: Atomic ({constraint_aware})")
        
        report_lines.append(f"\nHard Strategies (Level 2+): {len(compositionality_analysis['hard_strategies'])}")
        for name, info in compositionality_analysis['hard_strategies'].items():
            constraint_aware = "Constraint-aware" if info.get('constraint_aware', False) else "Basic"
            if info['composite']:
                components = ', '.join(info['composed_of']) if info['composed_of'] else 'None'
                report_lines.append(f"  - {name}: Composed of [{components}] ({constraint_aware})")
            else:
                report_lines.append(f"  - {name}: Atomic ({constraint_aware})")
        
        report_lines.append("")
        
        # Analyze each dataset
        all_stats = {}
        for dataset_file in dataset_files:
            try:
                dataset = self.load_dataset(dataset_file)
                stats = self.analyze_dataset_statistics(dataset)
                all_stats[dataset_file] = stats
                
                difficulty = dataset[0]['difficulty'] if dataset else 'unknown'
                puzzle_size = dataset[0]['size'] if dataset else 'unknown'
                
                report_lines.append(f"2. DATASET ANALYSIS: {dataset_file.upper()}")
                report_lines.append("-" * 40)
                report_lines.append(f"Difficulty Level: {difficulty}")
                report_lines.append(f"Puzzle Size: {puzzle_size}x{puzzle_size}")
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
                    for diff, avg_constraints in metrics['avg_constraints'].items():
                        report_lines.append(f"  Average constraints in {diff} puzzles: {avg_constraints:.1f}")
                    report_lines.append("")
                
                # Constraint analysis
                if 'constraint_analysis' in stats:
                    constraint_stats = stats['constraint_analysis']
                    report_lines.append("Constraint Analysis:")
                    for diff, density in constraint_stats['constraint_density_by_difficulty'].items():
                        report_lines.append(f"  {diff.capitalize()} puzzles:")
                        report_lines.append(f"    Horizontal constraints: {density['avg_horizontal']:.1f}")
                        report_lines.append(f"    Vertical constraints: {density['avg_vertical']:.1f}")
                        report_lines.append(f"    Total constraints: {density['avg_total']:.1f}")
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
        
        # Constraint-aware strategy analysis
        constraint_aware_count = 0
        total_strategy_instances = 0
        for stats in all_stats.values():
            for strategy, count in stats['strategy_usage'].items():
                total_strategy_instances += count
                if self.is_constraint_aware_strategy(strategy):
                    constraint_aware_count += count
        
        if total_strategy_instances > 0:
            constraint_aware_percentage = (constraint_aware_count / total_strategy_instances) * 100
            report_lines.append(f"Constraint-aware strategy usage: {constraint_aware_percentage:.1f}%")
        
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
                    prerequisites = info.get('prerequisites', [])
                    for prereq in prerequisites:
                        if prereq not in easy_strategy_names and prereq not in moderate_strategies:
                            print(f"Compositionality error: {name} requires unknown prerequisite {prereq}")
                            return False
            
            # Check that hard strategies compose easy/moderate strategies
            hard_strategies = self.hard_kb.get_all_strategies()
            moderate_strategy_names = set(self.moderate_kb.list_strategies())
            
            for name, info in hard_strategies.items():
                if info.get('composite', False):
                    prerequisites = info.get('prerequisites', [])
                    for prereq in prerequisites:
                        if (prereq not in easy_strategy_names and 
                            prereq not in moderate_strategy_names and 
                            prereq not in hard_strategies):
                            print(f"Compositionality error: {name} requires unknown prerequisite {prereq}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying compositionality: {e}")
            return False
    
    def export_strategy_graph(self, output_file: str = "futoshiki_strategy_composition_graph.json"):
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
                'description': info.get('description', ''),
                'applies_to': info.get('applies_to', [])
            })
            graph['levels']['easy'].append(name)
        
        # Add moderate strategy nodes and edges
        for name, info in self.moderate_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'moderate',
                'composite': info.get('composite', False),
                'description': info.get('description', ''),
                'constraint_aware': info.get('constraint_aware', False),
                'applies_to': info.get('applies_to', [])
            })
            graph['levels']['moderate'].append(name)
            
            # Add composition edges
            if info.get('composite', False):
                for prereq in info.get('prerequisites', []):
                    graph['edges'].append({
                        'source': prereq,
                        'target': name,
                        'type': 'prerequisite'
                    })
        
        # Add hard strategy nodes and edges
        for name, info in self.hard_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'hard',
                'composite': info.get('composite', False),
                'description': info.get('description', ''),
                'constraint_aware': info.get('constraint_aware', False),
                'applies_to': info.get('applies_to', [])
            })
            graph['levels']['hard'].append(name)
            
            # Add composition edges
            if info.get('composite', False):
                for prereq in info.get('prerequisites', []):
                    graph['edges'].append({
                        'source': prereq,
                        'target': name,
                        'type': 'prerequisite'
                    })
        
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        print(f"Strategy composition graph exported to {output_file}")
        return graph


def main():
    """Main function for dataset analysis"""
    analyzer = FutoshikiDatasetAnalyzer()
    
    # Define dataset files to analyze
    dataset_files = [
        "futoshiki_dataset_easy.json",
        "futoshiki_dataset_moderate.json",
        "futoshiki_dataset_hard.json"
    ]
    
    print("Starting Futoshiki dataset analysis...")
    
    # Generate comprehensive report
    report = analyzer.generate_report(dataset_files)
    
    # Generate visualizations for each dataset
    for dataset_file in dataset_files:
        try:
            dataset = analyzer.load_dataset(dataset_file)
            stats = analyzer.analyze_dataset_statistics(dataset)
            
            # Create output directory for this dataset
            difficulty = dataset[0]['difficulty'] if dataset else 'unknown'
            output_dir = f"futoshiki_analysis_plots_{difficulty}"
            
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
    print("FUTOSHIKI ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- futoshiki_analysis_report.txt")
    print("- futoshiki_strategy_composition_graph.json")
    print("- futoshiki_analysis_plots_[difficulty]/ directories")
    print("\nCompositionality verification:", 
          "PASSED" if analyzer.verify_compositionality() else "FAILED")

if __name__ == "__main__":
    main()