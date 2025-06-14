# kenken_analyzer.py
"""
KenKen Dataset Analyzer and Validator
Analyzes generated KenKen datasets for compositionality and strategy usage
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd

from kenken_easy_strategies_kb import EasyKenKenStrategiesKB
from kenken_moderate_strategies_kb import ModerateKenKenStrategiesKB
from kenken_hard_strategies_kb import HardKenKenStrategiesKB

class KenKenDatasetAnalyzer:
    def __init__(self):
        self.easy_kb = EasyKenKenStrategiesKB()
        self.moderate_kb = ModerateKenKenStrategiesKB()
        self.hard_kb = HardKenKenStrategiesKB()
    
    def load_dataset(self, filename: str) -> List[Dict]:
        """Load KenKen dataset from JSON file"""
        with open(filename, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def analyze_compositionality(self) -> Dict:
        """Analyze the compositionality relationships between KenKen strategies"""
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
                    if (component not in easy_strategies and 
                        component not in moderate_strategies and 
                        component not in hard_strategies):
                        print(f"Warning: {strategy_name} uses unknown component: {component}")
        
        return analysis
    
    def analyze_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Analyze statistics of the generated KenKen dataset"""
        stats = {
            'total_puzzles': len(dataset),
            'difficulty_distribution': Counter(),
            'strategy_usage': Counter(),
            'strategy_combinations': Counter(),
            'cage_analysis': {},
            'complexity_metrics': {},
            'validation_results': {}
        }
        
        # Basic statistics
        for puzzle in dataset:
            difficulty = puzzle['difficulty']
            strategies = puzzle['required_strategies']
            cages = puzzle['cages']
            
            stats['difficulty_distribution'][difficulty] += 1
            
            for strategy in strategies:
                stats['strategy_usage'][strategy] += 1
            
            # Strategy combinations
            strategy_combo = tuple(sorted(strategies))
            stats['strategy_combinations'][strategy_combo] += 1
        
        # Cage analysis
        stats['cage_analysis'] = self.analyze_cages(dataset)
        
        # Complexity metrics
        stats['complexity_metrics'] = self.calculate_complexity_metrics(dataset)
        
        # Validation
        stats['validation_results'] = self.validate_dataset(dataset)
        
        return stats
    
    def analyze_cages(self, dataset: List[Dict]) -> Dict:
        """Analyze cage characteristics in the dataset"""
        cage_stats = {
            'operation_distribution': Counter(),
            'size_distribution': Counter(),
            'target_distribution': defaultdict(list),
            'avg_cages_per_puzzle': 0,
            'cage_complexity': []
        }
        
        total_cages = 0
        
        for puzzle in dataset:
            cages = puzzle['cages']
            total_cages += len(cages)
            
            for cage in cages:
                operation = cage['operation']
                size = len(cage['cells'])
                target = cage['target']
                
                cage_stats['operation_distribution'][operation] += 1
                cage_stats['size_distribution'][size] += 1
                cage_stats['target_distribution'][operation].append(target)
                
                # Calculate cage complexity
                complexity = self.calculate_cage_complexity(cage)
                cage_stats['cage_complexity'].append(complexity)
        
        cage_stats['avg_cages_per_puzzle'] = total_cages / len(dataset) if dataset else 0
        cage_stats['avg_cage_complexity'] = np.mean(cage_stats['cage_complexity']) if cage_stats['cage_complexity'] else 0
        
        return cage_stats
    
    def calculate_cage_complexity(self, cage: Dict) -> float:
        """Calculate complexity score for a single cage"""
        size = len(cage['cells'])
        operation = cage['operation']
        target = cage['target']
        
        # Base complexity from size
        complexity = size * 0.5
        
        # Operation complexity
        op_complexity = {
            'single': 0.1,
            'addition': 0.3,
            'subtraction': 0.5,
            'multiplication': 0.7,
            'division': 0.9
        }
        complexity += op_complexity.get(operation, 0.5)
        
        # Target complexity
        if operation in ['multiplication', 'addition']:
            complexity += min(target / 20.0, 2.0)
        
        return complexity
    
    def calculate_complexity_metrics(self, dataset: List[Dict]) -> Dict:
        """Calculate complexity metrics for KenKen puzzles"""
        metrics = {
            'avg_strategies_per_puzzle': {},
            'avg_filled_cells': {},
            'avg_cages_per_puzzle': {},
            'strategy_depth_analysis': {},
            'cage_complexity_by_difficulty': {}
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
            
            # Average cages per puzzle
            cage_counts = [p['metadata']['total_cages'] for p in puzzles]
            metrics['avg_cages_per_puzzle'][difficulty] = np.mean(cage_counts)
            
            # Strategy depth analysis
            metrics['strategy_depth_analysis'][difficulty] = self.analyze_strategy_depth(puzzles)
            
            # Cage complexity by difficulty
            cage_complexities = []
            for puzzle in puzzles:
                for cage in puzzle['cages']:
                    cage_complexities.append(self.calculate_cage_complexity(cage))
            metrics['cage_complexity_by_difficulty'][difficulty] = np.mean(cage_complexities) if cage_complexities else 0
        
        return metrics
    
    def analyze_strategy_depth(self, puzzles: List[Dict]) -> Dict:
        """Analyze the depth of strategy composition for KenKen"""
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
        """Get the composition depth of a KenKen strategy"""
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
        """Validate KenKen puzzles in the dataset"""
        validation = {
            'valid_puzzles': 0,
            'invalid_puzzles': 0,
            'cage_validation': {},
            'grid_validation': {},
            'errors': []
        }
        
        for i, puzzle in enumerate(dataset):
            try:
                # Check puzzle format
                puzzle_grid = np.array(puzzle['puzzle_grid'])
                solution_grid = np.array(puzzle['solution_grid'])
                cages = puzzle['cages']
                grid_size = puzzle['grid_size']
                
                if puzzle_grid.shape != (grid_size, grid_size) or solution_grid.shape != (grid_size, grid_size):
                    validation['errors'].append(f"Puzzle {i}: Invalid grid dimensions")
                    validation['invalid_puzzles'] += 1
                    continue
                
                # Check solution correctness (Latin square property)
                if self.validate_kenken_solution(solution_grid):
                    validation['grid_validation'][f"puzzle_{i}"] = True
                else:
                    validation['grid_validation'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Invalid Latin square solution")
                
                # Check cage constraints
                if self.validate_all_cages(solution_grid, cages):
                    validation['cage_validation'][f"puzzle_{i}"] = True
                    validation['valid_puzzles'] += 1
                else:
                    validation['cage_validation'][f"puzzle_{i}"] = False
                    validation['errors'].append(f"Puzzle {i}: Cage constraints not satisfied")
                    validation['invalid_puzzles'] += 1
                
            except Exception as e:
                validation['errors'].append(f"Puzzle {i}: Error during validation - {str(e)}")
                validation['invalid_puzzles'] += 1
        
        return validation
    
    def validate_kenken_solution(self, grid: np.ndarray) -> bool:
        """Validate that grid satisfies KenKen Latin square constraints"""
        size = grid.shape[0]
        
        # Check rows
        for row in range(size):
            if set(grid[row, :]) != set(range(1, size + 1)):
                return False
        
        # Check columns
        for col in range(size):
            if set(grid[:, col]) != set(range(1, size + 1)):
                return False
        
        return True
    
    def validate_all_cages(self, grid: np.ndarray, cages: List[Dict]) -> bool:
        """Validate all cage arithmetic constraints"""
        for cage in cages:
            if not self.validate_cage_constraint(grid, cage):
                return False
        return True
    
    def validate_cage_constraint(self, grid: np.ndarray, cage: Dict) -> bool:
        """Validate a single cage constraint"""
        cells = cage['cells']
        operation = cage['operation']
        target = cage['target']
        
        values = [grid[r, c] for r, c in cells]
        
        if operation == 'single':
            return len(values) == 1 and values[0] == target
        elif operation == 'addition':
            return sum(values) == target
        elif operation == 'subtraction':
            if len(values) == 2:
                return abs(values[0] - values[1]) == target
            return False
        elif operation == 'multiplication':
            result = 1
            for v in values:
                result *= v
            return result == target
        elif operation == 'division':
            if len(values) == 2:
                return (values[0] / values[1] == target or values[1] / values[0] == target)
            return False
        
        return False
    
    def generate_visualizations(self, stats: Dict, output_dir: str = "kenken_analysis_plots"):
        """Generate visualization plots for the KenKen analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy usage distribution
        plt.figure(figsize=(12, 8))
        strategies = list(stats['strategy_usage'].keys())
        counts = list(stats['strategy_usage'].values())
        
        plt.barh(strategies, counts)
        plt.title('KenKen Strategy Usage Distribution')
        plt.xlabel('Number of Puzzles')
        plt.ylabel('Strategy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kenken_strategy_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Difficulty distribution
        plt.figure(figsize=(8, 6))
        difficulties = list(stats['difficulty_distribution'].keys())
        counts = list(stats['difficulty_distribution'].values())
        
        plt.pie(counts, labels=difficulties, autopct='%1.1f%%')
        plt.title('KenKen Difficulty Distribution')
        plt.savefig(f"{output_dir}/kenken_difficulty_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cage operation distribution
        if 'cage_analysis' in stats:
            cage_stats = stats['cage_analysis']
            
            plt.figure(figsize=(10, 6))
            operations = list(cage_stats['operation_distribution'].keys())
            op_counts = list(cage_stats['operation_distribution'].values())
            
            plt.bar(operations, op_counts)
            plt.title('KenKen Cage Operation Distribution')
            plt.xlabel('Operation')
            plt.ylabel('Number of Cages')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/kenken_cage_operations.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Cage size distribution
            plt.figure(figsize=(8, 6))
            sizes = list(cage_stats['size_distribution'].keys())
            size_counts = list(cage_stats['size_distribution'].values())
            
            plt.bar(sizes, size_counts)
            plt.title('KenKen Cage Size Distribution')
            plt.xlabel('Cage Size (Number of Cells)')
            plt.ylabel('Number of Cages')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/kenken_cage_sizes.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Complexity metrics
        if 'complexity_metrics' in stats:
            metrics = stats['complexity_metrics']
            
            # Strategies per puzzle by difficulty
            if 'avg_strategies_per_puzzle' in metrics:
                plt.figure(figsize=(10, 6))
                difficulties = list(metrics['avg_strategies_per_puzzle'].keys())
                avg_strategies = list(metrics['avg_strategies_per_puzzle'].values())
                
                plt.bar(difficulties, avg_strategies)
                plt.title('Average KenKen Strategies per Puzzle by Difficulty')
                plt.xlabel('Difficulty')
                plt.ylabel('Average Number of Strategies')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/kenken_avg_strategies_by_difficulty.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Cage complexity by difficulty
            if 'cage_complexity_by_difficulty' in metrics:
                plt.figure(figsize=(10, 6))
                difficulties = list(metrics['cage_complexity_by_difficulty'].keys())
                complexities = list(metrics['cage_complexity_by_difficulty'].values())
                
                plt.bar(difficulties, complexities)
                plt.title('Average KenKen Cage Complexity by Difficulty')
                plt.xlabel('Difficulty')
                plt.ylabel('Average Cage Complexity Score')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/kenken_cage_complexity_by_difficulty.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, dataset_files: List[str], output_file: str = "kenken_analysis_report.txt"):
        """Generate a comprehensive KenKen analysis report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MNIST KENKEN DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Analyze compositionality
        compositionality_analysis = self.analyze_compositionality()
        report_lines.append("1. KENKEN COMPOSITIONALITY ANALYSIS")
        report_lines.append("-" * 40)
        
        report_lines.append(f"Easy KenKen Strategies (Base Level): {len(compositionality_analysis['easy_strategies'])}")
        for name, info in compositionality_analysis['easy_strategies'].items():
            report_lines.append(f"  - {name}: {'Composite' if info['composite'] else 'Atomic'}")
        
        report_lines.append(f"\nModerate KenKen Strategies (Level 1): {len(compositionality_analysis['moderate_strategies'])}")
        for name, info in compositionality_analysis['moderate_strategies'].items():
            if info['composite']:
                components = ', '.join(info['composed_of'])
                report_lines.append(f"  - {name}: Composed of [{components}]")
            else:
                report_lines.append(f"  - {name}: Atomic")
        
        report_lines.append(f"\nHard KenKen Strategies (Level 2+): {len(compositionality_analysis['hard_strategies'])}")
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
                
                report_lines.append(f"2. KENKEN DATASET ANALYSIS: {dataset_file.upper()}")
                report_lines.append("-" * 40)
                report_lines.append(f"Difficulty Level: {difficulty}")
                report_lines.append(f"Total Puzzles: {stats['total_puzzles']}")
                
                if dataset:
                    grid_size = dataset[0]['grid_size']
                    report_lines.append(f"Grid Size: {grid_size}x{grid_size}")
                
                report_lines.append("")
                
                # Strategy usage
                report_lines.append("Strategy Usage:")
                for strategy, count in stats['strategy_usage'].most_common():
                    percentage = (count / stats['total_puzzles']) * 100
                    report_lines.append(f"  {strategy}: {count} ({percentage:.1f}%)")
                report_lines.append("")
                
                # Cage analysis
                if 'cage_analysis' in stats:
                    cage_stats = stats['cage_analysis']
                    report_lines.append("Cage Analysis:")
                    report_lines.append(f"  Average cages per puzzle: {cage_stats['avg_cages_per_puzzle']:.1f}")
                    report_lines.append(f"  Average cage complexity: {cage_stats['avg_cage_complexity']:.2f}")
                    
                    report_lines.append("  Operation distribution:")
                    for op, count in cage_stats['operation_distribution'].items():
                        report_lines.append(f"    {op}: {count}")
                    
                    report_lines.append("  Size distribution:")
                    for size, count in cage_stats['size_distribution'].items():
                        report_lines.append(f"    {size} cells: {count}")
                    report_lines.append("")
                
                # Complexity metrics
                if 'complexity_metrics' in stats:
                    metrics = stats['complexity_metrics']
                    report_lines.append("Complexity Metrics:")
                    for diff, avg_strategies in metrics['avg_strategies_per_puzzle'].items():
                        report_lines.append(f"  Average strategies per {diff} puzzle: {avg_strategies:.2f}")
                    for diff, avg_filled in metrics['avg_filled_cells'].items():
                        report_lines.append(f"  Average filled cells in {diff} puzzles: {avg_filled:.1f}")
                    for diff, avg_cages in metrics['avg_cages_per_puzzle'].items():
                        report_lines.append(f"  Average cages in {diff} puzzles: {avg_cages:.1f}")
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
                    recommendations.append(f"Review KenKen puzzle generation logic for {filename}")
            
            # Check strategy distribution
            strategy_counts = list(stats['strategy_usage'].values())
            if len(strategy_counts) > 0:
                min_usage = min(strategy_counts)
                max_usage = max(strategy_counts)
                if max_usage > min_usage * 5:  # High imbalance
                    issues_found.append(f"Unbalanced strategy usage in {filename}")
                    recommendations.append(f"Balance strategy selection in {filename}")
            
            # Check cage distribution
            cage_stats = stats.get('cage_analysis', {})
            if 'operation_distribution' in cage_stats:
                op_counts = list(cage_stats['operation_distribution'].values())
                if len(op_counts) > 1:
                    min_op = min(op_counts)
                    max_op = max(op_counts)
                    if max_op > min_op * 10:  # Very unbalanced
                        issues_found.append(f"Unbalanced cage operations in {filename}")
                        recommendations.append(f"Diversify cage operation types in {filename}")
        
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
            report_lines.append("No significant issues found. KenKen dataset appears well-balanced.")
        
        report_lines.append("")
        report_lines.append("5. KENKEN DATASET SUMMARY")
        report_lines.append("-" * 40)
        
        total_puzzles = sum(stats['total_puzzles'] for stats in all_stats.values())
        total_strategies = len(set().union(*[stats['strategy_usage'].keys() for stats in all_stats.values()]))
        
        report_lines.append(f"Total KenKen puzzles across all datasets: {total_puzzles}")
        report_lines.append(f"Total unique strategies used: {total_strategies}")
        report_lines.append(f"Datasets analyzed: {len(all_stats)}")
        
        # Compositionality verification
        composition_verified = self.verify_compositionality()
        report_lines.append(f"Compositionality structure verified: {'Yes' if composition_verified else 'No'}")
        
        # KenKen-specific metrics
        if all_stats:
            total_cages = 0
            all_operations = []
            
            for stats in all_stats.values():
                cage_stats = stats.get('cage_analysis', {})
                if 'operation_distribution' in cage_stats:
                    all_operations.extend(cage_stats['operation_distribution'].keys())
                    total_cages += sum(cage_stats['operation_distribution'].values())
            
            unique_operations = set(all_operations)
            report_lines.append(f"Total cages across all puzzles: {total_cages}")
            report_lines.append(f"Unique cage operations: {', '.join(unique_operations)}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"KenKen analysis report saved to {output_file}")
        return '\n'.join(report_lines)
    
    def verify_compositionality(self) -> bool:
        """Verify that the KenKen compositionality structure is correct"""
        try:
            # Check that moderate strategies compose easy strategies
            moderate_strategies = self.moderate_kb.get_all_strategies()
            easy_strategy_names = set(self.easy_kb.list_strategies())
            
            for name, info in moderate_strategies.items():
                if info.get('composite', False):
                    composed_of = info.get('composed_of', [])
                    for component in composed_of:
                        if component not in easy_strategy_names and component not in moderate_strategies:
                            print(f"KenKen Compositionality error: {name} uses unknown component {component}")
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
                            print(f"KenKen Compositionality error: {name} uses unknown component {component}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying KenKen compositionality: {e}")
            return False
    
    def export_strategy_graph(self, output_file: str = "kenken_strategy_composition_graph.json"):
        """Export KenKen strategy composition as a graph structure"""
        graph = {
            'nodes': [],
            'edges': [],
            'levels': {
                'easy': [],
                'moderate': [],
                'hard': []
            },
            'puzzle_type': 'kenken'
        }
        
        # Add easy strategy nodes
        for name, info in self.easy_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'easy',
                'composite': info.get('composite', False),
                'description': info.get('description', ''),
                'puzzle_type': 'kenken'
            })
            graph['levels']['easy'].append(name)
        
        # Add moderate strategy nodes and edges
        for name, info in self.moderate_kb.get_all_strategies().items():
            graph['nodes'].append({
                'id': name,
                'level': 'moderate',
                'composite': info.get('composite', False),
                'description': info.get('description', ''),
                'puzzle_type': 'kenken'
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
                'description': info.get('description', ''),
                'puzzle_type': 'kenken'
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
        
        print(f"KenKen strategy composition graph exported to {output_file}")
        return graph


def main():
    """Main function for KenKen dataset analysis"""
    analyzer = KenKenDatasetAnalyzer()
    
    # Define dataset files to analyze
    dataset_files = [
        "kenken_dataset_easy.json",
        "kenken_dataset_moderate.json",
        "kenken_dataset_hard.json"
    ]
    
    print("Starting KenKen dataset analysis...")
    
    # Generate comprehensive report
    report = analyzer.generate_report(dataset_files)
    
    # Generate visualizations for each dataset
    for dataset_file in dataset_files:
        try:
            dataset = analyzer.load_dataset(dataset_file)
            stats = analyzer.analyze_dataset_statistics(dataset)
            
            # Create output directory for this dataset
            difficulty = dataset[0]['difficulty'] if dataset else 'unknown'
            output_dir = f"kenken_analysis_plots_{difficulty}"
            
            analyzer.generate_visualizations(stats, output_dir)
            print(f"KenKen visualizations saved to {output_dir}/")
            
        except FileNotFoundError:
            print(f"Dataset file {dataset_file} not found, skipping...")
        except Exception as e:
            print(f"Error analyzing {dataset_file}: {e}")
    
    # Export strategy composition graph
    analyzer.export_strategy_graph()
    
    # Print summary
    print("\n" + "="*60)
    print("KENKEN ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- kenken_analysis_report.txt")
    print("- kenken_strategy_composition_graph.json")
    print("- kenken_analysis_plots_[difficulty]/ directories")
    print("\nKenKen Compositionality verification:", 
          "PASSED" if analyzer.verify_compositionality() else "FAILED")


if __name__ == "__main__":
    main()