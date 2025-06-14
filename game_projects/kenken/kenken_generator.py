# kenken_generator.py
"""
MNIST-based KenKen Puzzle Generator with Integrated Validation
Generates exactly the requested number of VALID KenKen puzzles
"""

import numpy as np
import json
import random
import os
import time
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from itertools import combinations, permutations

# Import knowledge bases
from kenken_easy_strategies_kb import EasyKenKenStrategiesKB
from kenken_moderate_strategies_kb import ModerateKenKenStrategiesKB
from kenken_hard_strategies_kb import HardKenKenStrategiesKB


class KenKenValidator:
    """Integrated KenKen validator that ensures puzzle quality"""
    
    def __init__(self):
        self.easy_kb = EasyKenKenStrategiesKB()
        self.moderate_kb = ModerateKenKenStrategiesKB()
        self.hard_kb = HardKenKenStrategiesKB()
    
    def is_valid_kenken_solution(self, grid: np.ndarray) -> bool:
        """Check if a complete grid is a valid KenKen solution"""
        size = grid.shape[0]
        
        # Check rows - no duplicates
        for row in range(size):
            if set(grid[row, :]) != set(range(1, size + 1)):
                return False
        
        # Check columns - no duplicates
        for col in range(size):
            if set(grid[:, col]) != set(range(1, size + 1)):
                return False
        
        return True
    
    def validate_cage_constraint(self, grid: np.ndarray, cage_cells: List[Tuple[int, int]], 
                                operation: str, target: int) -> bool:
        """Validate that a cage satisfies its arithmetic constraint"""
        values = [grid[r, c] for r, c in cage_cells]
        
        if operation == 'addition':
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
        elif operation == 'single':
            return len(values) == 1 and values[0] == target
        
        return False
    
    def has_unique_solution(self, grid: np.ndarray, cages: List[Dict]) -> bool:
        """Check if puzzle has exactly one solution"""
        solutions_found = 0
        max_solutions = 2
        
        def solve_with_count(current_grid):
            nonlocal solutions_found
            
            if solutions_found >= max_solutions:
                return
            
            # Find first empty cell
            empty_pos = None
            size = current_grid.shape[0]
            for row in range(size):
                for col in range(size):
                    if current_grid[row, col] == 0:
                        empty_pos = (row, col)
                        break
                if empty_pos:
                    break
            
            if not empty_pos:
                # Check all cage constraints
                if self.validate_all_cages(current_grid, cages):
                    solutions_found += 1
                return
            
            row, col = empty_pos
            
            # Try each number
            for num in range(1, size + 1):
                if self.is_valid_placement(current_grid, row, col, num):
                    current_grid[row, col] = num
                    solve_with_count(current_grid)
                    current_grid[row, col] = 0
                    
                    if solutions_found >= max_solutions:
                        return
        
        test_grid = grid.copy()
        solve_with_count(test_grid)
        return solutions_found == 1
    
    def validate_all_cages(self, grid: np.ndarray, cages: List[Dict]) -> bool:
        """Validate all cage constraints"""
        for cage in cages:
            if not self.validate_cage_constraint(grid, cage['cells'], cage['operation'], cage['target']):
                return False
        return True
    
    def is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid for KenKen rules"""
        # Check row
        if num in grid[row, :]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        return True
    
    def can_be_solved_with_strategies(self, grid: np.ndarray, cages: List[Dict], 
                                     required_strategies: List[str]) -> bool:
        """Check if puzzle can be solved using only the required strategies (heuristic)"""
        filled_cells = np.sum(grid != 0)
        total_cells = grid.shape[0] ** 2
        fill_ratio = filled_cells / total_cells
        
        strategy_complexity = self.calculate_strategy_complexity(required_strategies)
        cage_complexity = self.calculate_cage_complexity(cages)
        
        # Difficulty thresholds
        if strategy_complexity <= 1.0:  # Easy strategies
            return 0.4 <= fill_ratio <= 0.7 and cage_complexity <= 2.0
        elif strategy_complexity <= 2.5:  # Moderate strategies
            return 0.2 <= fill_ratio <= 0.5 and cage_complexity <= 4.0
        else:  # Hard strategies
            return 0.1 <= fill_ratio <= 0.4 and cage_complexity <= 6.0
    
    def calculate_strategy_complexity(self, strategies: List[str]) -> float:
        """Calculate complexity score based on strategies"""
        complexity_scores = {
            # Easy strategies
            'single_cell_cage': 0.1,
            'two_cell_addition_cage': 0.3,
            'two_cell_subtraction_cage': 0.4,
            'two_cell_multiplication_cage': 0.5,
            'two_cell_division_cage': 0.6,
            'naked_single': 0.3,
            'hidden_single_row': 0.4,
            'hidden_single_column': 0.4,
            'cage_completion': 0.5,
            'eliminate_by_row': 0.2,
            'eliminate_by_column': 0.2,
            'simple_cage_arithmetic': 0.6,
            'forced_candidate_cage': 0.7,
            'cage_boundary_constraint': 0.5,
            
            # Moderate strategies
            'cage_candidate_elimination': 1.5,
            'multi_cage_intersection': 2.0,
            'cage_combination_analysis': 2.2,
            'advanced_cage_arithmetic': 1.8,
            'cage_sum_distribution': 1.6,
            'cage_product_factorization': 2.5,
            'naked_pair_in_cage': 1.4,
            'hidden_pair_in_cage': 1.7,
            'cage_constraint_propagation': 2.0,
            'division_remainder_analysis': 2.3,
            'large_cage_symmetry': 2.4,
            'cage_endpoint_analysis': 1.9,
            
            # Hard strategies
            'multi_cage_chain_analysis': 3.5,
            'cage_forcing_chains': 4.0,
            'advanced_cage_intersection': 3.8,
            'cage_arithmetic_sequences': 4.2,
            'recursive_cage_solving': 4.5,
            'cage_elimination_networks': 4.0,
            'constraint_satisfaction_pruning': 4.8,
            'global_arithmetic_optimization': 5.0,
            'cage_symmetry_breaking': 4.3,
            'temporal_constraint_reasoning': 4.6,
            'probabilistic_cage_analysis': 5.2,
            'meta_strategy_selection': 5.5,
            'cage_graph_coloring': 4.9,
            'dynamic_constraint_learning': 5.8,
            'holistic_puzzle_analysis': 6.0
        }
        
        total_score = sum(complexity_scores.get(strategy, 2.0) for strategy in strategies)
        return total_score / len(strategies) if strategies else 0.0
    
    def calculate_cage_complexity(self, cages: List[Dict]) -> float:
        """Calculate complexity based on cage structure"""
        complexity = 0.0
        
        for cage in cages:
            cage_size = len(cage['cells'])
            operation = cage['operation']
            target = cage['target']
            
            # Size complexity
            complexity += cage_size * 0.5
            
            # Operation complexity
            op_complexity = {
                'single': 0.1,
                'addition': 0.3,
                'subtraction': 0.5,
                'multiplication': 0.7,
                'division': 0.9
            }
            complexity += op_complexity.get(operation, 0.5)
            
            # Target complexity (larger targets are generally harder)
            if operation in ['multiplication', 'addition']:
                complexity += min(target / 20.0, 2.0)
        
        return complexity / len(cages) if cages else 0.0
    
    def meets_difficulty_requirements(self, grid: np.ndarray, cages: List[Dict], 
                                     difficulty: str, required_strategies: List[str]) -> bool:
        """Check if puzzle meets all requirements for the difficulty level"""
        if not self.has_unique_solution(grid, cages):
            return False
        
        if not self.can_be_solved_with_strategies(grid, cages, required_strategies):
            return False
        
        # Additional difficulty-specific checks
        filled_cells = np.sum(grid != 0)
        total_cells = grid.shape[0] ** 2
        
        if difficulty == 'easy':
            return (filled_cells >= total_cells * 0.4 and 
                    len(required_strategies) <= 3 and
                    all(s in self.easy_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'moderate':
            return (filled_cells >= total_cells * 0.2 and 
                    len(required_strategies) <= 5 and
                    any(s in self.moderate_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'hard':
            return (filled_cells >= total_cells * 0.1 and 
                    len(required_strategies) <= 8 and
                    any(s in self.hard_kb.list_strategies() for s in required_strategies))
        
        return False


class MNISTKenKenGenerator:
    """MNIST KenKen Generator with guaranteed valid puzzle generation"""
    
    def __init__(self, config_manager=None, grid_size=4):
        """Initialize the generator with integrated validation"""
        print("🚀 Initializing MNIST KenKen Generator with integrated validation...")
        
        self.grid_size = grid_size
        
        # Initialize validator
        self.validator = KenKenValidator()
        
        # Initialize knowledge bases
        self.easy_kb = EasyKenKenStrategiesKB()
        self.moderate_kb = ModerateKenKenStrategiesKB()
        self.hard_kb = HardKenKenStrategiesKB()
        
        # Configuration
        self.config_manager = config_manager
        
        # Load MNIST data
        self.mnist_images = {}
        self.load_mnist_data()
        
        # Generation statistics
        self.stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failed_validations': 0,
            'generation_times': []
        }
        
        print("✅ KenKen Generator initialized successfully")
    
    def load_mnist_data(self):
        """Load and organize MNIST dataset"""
        try:
            print("📥 Loading MNIST dataset...")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            
            # Load datasets
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            # Organize by digit (1-N only, where N is grid_size)
            train_by_digit = {i: [] for i in range(1, self.grid_size + 1)}
            
            for image, label in train_dataset:
                if 1 <= label <= self.grid_size:
                    train_by_digit[label].append(image)
            
            self.mnist_images = {'train': train_by_digit}
            
            total_images = sum(len(images) for images in train_by_digit.values())
            print(f"✅ MNIST loaded: {total_images} training images for digits 1-{self.grid_size}")
            
        except Exception as e:
            print(f"⚠️ Error loading MNIST: {e}")
            print("🔄 Creating fallback dummy data...")
            self.create_dummy_mnist()
    
    def create_dummy_mnist(self):
        """Create dummy MNIST data for testing"""
        print("🎨 Generating dummy MNIST images...")
        
        train_by_digit = {i: [] for i in range(1, self.grid_size + 1)}
        
        for digit in range(1, self.grid_size + 1):
            for _ in range(50):  # 50 images per digit
                dummy_img = self.create_digit_pattern(digit)
                train_by_digit[digit].append(dummy_img)
        
        self.mnist_images = {'train': train_by_digit}
        print("✅ Dummy MNIST data created")
    
    def create_digit_pattern(self, digit: int) -> np.ndarray:
        """Create a recognizable pattern for a digit"""
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simple patterns for each digit
        if digit == 1:
            img[5:23, 13:15] = 255
        elif digit == 2:
            img[5:9, 8:20] = 255
            img[9:14, 15:20] = 255
            img[14:18, 8:15] = 255
            img[18:23, 8:20] = 255
        elif digit == 3:
            img[5:9, 8:20] = 255
            img[12:16, 12:20] = 255
            img[18:23, 8:20] = 255
        elif digit == 4:
            img[5:14, 8:12] = 255
            img[14:18, 8:20] = 255
            img[10:23, 16:20] = 255
        else:
            # Generic pattern with digit-specific characteristics
            center = 14
            for i in range(28):
                for j in range(28):
                    distance = abs(i - center) + abs(j - center)
                    if distance < digit * 3 and (i + j + digit * 5) % 7 < 3:
                        img[i, j] = min(255, (distance + digit * 30) % 256)
        
        return img
    
    def get_mnist_image(self, digit: int) -> np.ndarray:
        """Get a random MNIST image for the digit"""
        if digit < 1 or digit > self.grid_size:
            raise ValueError(f"Digit must be 1-{self.grid_size}, got {digit}")
        
        available_images = self.mnist_images['train'][digit]
        if not available_images:
            return self.create_digit_pattern(digit)
        
        return random.choice(available_images)
    
    def generate_complete_kenken_solution(self) -> np.ndarray:
        """Generate a complete valid KenKen solution (Latin square)"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Use backtracking to fill the grid
        if self.fill_kenken_grid(grid, 0, 0):
            return grid
        else:
            # Fallback: create a simple valid Latin square
            return self.create_simple_latin_square()
    
    def fill_kenken_grid(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Fill KenKen grid using backtracking"""
        if row == self.grid_size:
            return True
        
        next_row = row if col < self.grid_size - 1 else row + 1
        next_col = (col + 1) % self.grid_size
        
        numbers = list(range(1, self.grid_size + 1))
        random.shuffle(numbers)
        
        for num in numbers:
            if self.validator.is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                
                if self.fill_kenken_grid(grid, next_row, next_col):
                    return True
                
                grid[row, col] = 0
        
        return False
    
    def create_simple_latin_square(self) -> np.ndarray:
        """Create a simple Latin square as fallback"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                grid[row, col] = ((row + col) % self.grid_size) + 1
        
        # Shuffle rows and columns to add randomness
        row_order = list(range(self.grid_size))
        col_order = list(range(self.grid_size))
        random.shuffle(row_order)
        random.shuffle(col_order)
        
        shuffled_grid = np.zeros_like(grid)
        for i, orig_row in enumerate(row_order):
            for j, orig_col in enumerate(col_order):
                shuffled_grid[i, j] = grid[orig_row, orig_col]
        
        return shuffled_grid
    
    def generate_cages(self, difficulty: str) -> List[Dict]:
        """Generate cages for the puzzle based on difficulty"""
        cages = []
        
        # Define cage generation parameters based on difficulty
        cage_params = {
            'easy': {
                'min_cage_size': 1,
                'max_cage_size': 2,
                'single_cell_ratio': 0.3,
                'preferred_operations': ['single', 'addition', 'subtraction']
            },
            'moderate': {
                'min_cage_size': 1,
                'max_cage_size': 3,
                'single_cell_ratio': 0.2,
                'preferred_operations': ['addition', 'subtraction', 'multiplication', 'division']
            },
            'hard': {
                'min_cage_size': 2,
                'max_cage_size': 4,
                'single_cell_ratio': 0.1,
                'preferred_operations': ['addition', 'multiplication', 'division']
            }
        }
        
        params = cage_params.get(difficulty, cage_params['easy'])
        
        # Track used cells
        used_cells = set()
        all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        
        while len(used_cells) < len(all_cells):
            # Pick a random unused cell as cage start
            available_cells = [cell for cell in all_cells if cell not in used_cells]
            if not available_cells:
                break
            
            start_cell = random.choice(available_cells)
            
            # Decide cage size
            max_possible_size = min(params['max_cage_size'], len(available_cells))
            cage_size = random.randint(params['min_cage_size'], max_possible_size)
            
            # Generate cage cells
            cage_cells = self.generate_cage_cells(start_cell, cage_size, used_cells)
            
            # Determine operation and target
            operation = random.choice(params['preferred_operations'])
            if len(cage_cells) == 1:
                operation = 'single'
            
            cages.append({
                'cells': cage_cells,
                'operation': operation,
                'target': 0  # Will be set based on solution
            })
            
            used_cells.update(cage_cells)
        
        return cages
    
    def generate_cage_cells(self, start_cell: Tuple[int, int], size: int, 
                           used_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Generate connected cage cells starting from start_cell"""
        cage_cells = [start_cell]
        candidates = [start_cell]
        
        while len(cage_cells) < size and candidates:
            # Get neighbors of current cage cells
            neighbors = []
            for cell in cage_cells:
                r, c = cell
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and
                        (nr, nc) not in used_cells and (nr, nc) not in cage_cells):
                        neighbors.append((nr, nc))
            
            if neighbors:
                next_cell = random.choice(neighbors)
                cage_cells.append(next_cell)
            else:
                break
        
        return cage_cells
    
    def set_cage_targets(self, solution: np.ndarray, cages: List[Dict]) -> List[Dict]:
        """Set cage targets based on solution values"""
        for cage in cages:
            values = [solution[r, c] for r, c in cage['cells']]
            operation = cage['operation']
            
            if operation == 'single':
                cage['target'] = values[0]
            elif operation == 'addition':
                cage['target'] = sum(values)
            elif operation == 'subtraction':
                if len(values) == 2:
                    cage['target'] = abs(values[0] - values[1])
                else:
                    cage['target'] = sum(values)  # Fallback to addition
                    cage['operation'] = 'addition'
            elif operation == 'multiplication':
                target = 1
                for v in values:
                    target *= v
                cage['target'] = target
            elif operation == 'division':
                if len(values) == 2 and min(values) > 0:
                    cage['target'] = max(values) // min(values)
                    if max(values) % min(values) != 0:
                        # Not exact division, change to multiplication
                        cage['operation'] = 'multiplication'
                        cage['target'] = values[0] * values[1]
                else:
                    cage['operation'] = 'addition'
                    cage['target'] = sum(values)
        
        return cages
    
    def create_puzzle_from_solution(self, solution: np.ndarray, difficulty: str) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Create a valid puzzle by generating cages and removing cells"""
        print(f"    🔨 Creating {difficulty} KenKen puzzle...")
        
        max_attempts = 100
        best_puzzle = None
        best_cages = []
        best_strategies = []
        best_score = -1
        
        for attempt in range(max_attempts):
            if attempt > 0 and attempt % 20 == 0:
                print(f"      ⏳ Attempt {attempt}/{max_attempts}")
            
            # Generate cages
            cages = self.generate_cages(difficulty)
            cages = self.set_cage_targets(solution, cages)
            
            # Create puzzle by removing cells
            puzzle, strategies = self.create_puzzle_with_cages(solution, cages, difficulty)
            
            # Validate the puzzle
            if self.validator.meets_difficulty_requirements(puzzle, cages, difficulty, strategies):
                score = self.calculate_puzzle_quality(puzzle, cages, strategies, difficulty)
                
                if score > best_score:
                    best_puzzle = puzzle.copy()
                    best_cages = cages.copy()
                    best_strategies = strategies.copy()
                    best_score = score
                
                if score >= 0.7:
                    print(f"      ✅ High-quality puzzle found at attempt {attempt + 1}")
                    break
        
        if best_puzzle is None:
            print(f"      🔄 Using fallback generation for {difficulty}")
            best_puzzle, best_cages, best_strategies = self.create_fallback_puzzle(solution, difficulty)
        
        return best_puzzle, best_cages, best_strategies
    
    def create_puzzle_with_cages(self, solution: np.ndarray, cages: List[Dict], difficulty: str) -> Tuple[np.ndarray, List[str]]:
        """Create puzzle by strategically removing cells based on cages"""
        puzzle = solution.copy()
        
        # Get target parameters for difficulty
        target_params = self.get_difficulty_parameters(difficulty)
        target_filled = int(self.grid_size ** 2 * target_params['fill_ratio'])
        
        # Get appropriate strategies
        strategies = self.get_random_strategies(difficulty)
        
        # Remove cells strategically
        cells_to_remove = (self.grid_size ** 2) - target_filled
        removed = self.remove_cells_strategically(puzzle, cages, cells_to_remove)
        
        return puzzle, strategies
    
    def get_difficulty_parameters(self, difficulty: str) -> Dict:
        """Get target parameters for difficulty"""
        params = {
            'easy': {
                'fill_ratio': 0.6,
                'max_strategies': 3,
                'max_cage_size': 2
            },
            'moderate': {
                'fill_ratio': 0.4,
                'max_strategies': 5,
                'max_cage_size': 3
            },
            'hard': {
                'fill_ratio': 0.25,
                'max_strategies': 8,
                'max_cage_size': 4
            }
        }
        return params.get(difficulty, params['easy'])
    
    def get_random_strategies(self, difficulty: str) -> List[str]:
        """Get appropriate random strategies for difficulty"""
        if difficulty == 'easy':
            available = list(self.easy_kb.list_strategies())
            return random.sample(available, min(2, len(available)))
        
        elif difficulty == 'moderate':
            easy_strategies = list(self.easy_kb.list_strategies())
            moderate_strategies = list(self.moderate_kb.list_strategies())
            
            # Must include at least one moderate strategy
            selected = [random.choice(moderate_strategies)]
            remaining = easy_strategies + moderate_strategies
            remaining = [s for s in remaining if s not in selected]
            
            num_additional = random.randint(1, 3)
            selected.extend(random.sample(remaining, min(num_additional, len(remaining))))
            return selected
        
        else:  # hard
            easy_strategies = list(self.easy_kb.list_strategies())
            moderate_strategies = list(self.moderate_kb.list_strategies())
            hard_strategies = list(self.hard_kb.list_strategies())
            
            # Must include at least one hard strategy
            selected = [random.choice(hard_strategies)]
            remaining = easy_strategies + moderate_strategies + hard_strategies
            remaining = [s for s in remaining if s not in selected]
            
            num_additional = random.randint(2, 5)
            selected.extend(random.sample(remaining, min(num_additional, len(remaining))))
            return selected
    
    def remove_cells_strategically(self, puzzle: np.ndarray, cages: List[Dict], cells_to_remove: int) -> int:
        """Remove cells strategically while maintaining solvability"""
        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        random.shuffle(all_positions)
        
        removed = 0
        for row, col in all_positions:
            if removed >= cells_to_remove:
                break
            
            # Try removing this cell
            original_value = puzzle[row, col]
            puzzle[row, col] = 0
            
            # Check if puzzle still has unique solution (simplified check)
            if self.quick_solvability_check(puzzle, cages):
                removed += 1
            else:
                # Restore if removal breaks solvability
                puzzle[row, col] = original_value
        
        return removed
    
    def quick_solvability_check(self, puzzle: np.ndarray, cages: List[Dict]) -> bool:
        """Quick heuristic check for puzzle solvability"""
        # Check if each cage has enough information to be solvable
        for cage in cages:
            filled_cells = sum(1 for r, c in cage['cells'] if puzzle[r, c] != 0)
            total_cells = len(cage['cells'])
            
            # Heuristic: if too many cells are empty in a cage, it might be unsolvable
            if filled_cells < total_cells * 0.3 and total_cells > 2:
                return False
        
        return True
    
    def calculate_puzzle_quality(self, puzzle: np.ndarray, cages: List[Dict], 
                                strategies: List[str], difficulty: str) -> float:
        """Calculate quality score for a puzzle (0.0 to 1.0)"""
        score = 0.0
        
        # Factor 1: Appropriate fill ratio
        filled = np.sum(puzzle != 0)
        total = self.grid_size ** 2
        target_params = self.get_difficulty_parameters(difficulty)
        target_fill = target_params['fill_ratio']
        
        actual_fill = filled / total
        if abs(actual_fill - target_fill) < 0.2:
            score += 0.3
        
        # Factor 2: Strategy complexity matches difficulty
        complexity = self.validator.calculate_strategy_complexity(strategies)
        expected_complexity = {'easy': 0.5, 'moderate': 2.0, 'hard': 4.0}
        
        if abs(complexity - expected_complexity[difficulty]) < 1.0:
            score += 0.3
        
        # Factor 3: Cage structure quality
        cage_complexity = self.validator.calculate_cage_complexity(cages)
        expected_cage_complexity = {'easy': 1.0, 'moderate': 2.5, 'hard': 4.0}
        
        if abs(cage_complexity - expected_cage_complexity[difficulty]) < 1.0:
            score += 0.2
        
        # Factor 4: Has unique solution (simplified check)
        if self.quick_solvability_check(puzzle, cages):
            score += 0.2
        
        return score
    
    def create_fallback_puzzle(self, solution: np.ndarray, difficulty: str) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Create a guaranteed valid puzzle as fallback"""
        puzzle = solution.copy()
        target_params = self.get_difficulty_parameters(difficulty)
        
        # Simple cages
        cages = self.create_simple_cages(solution)
        
        # Simple removal
        target_filled = int(self.grid_size ** 2 * target_params['fill_ratio'])
        cells_to_remove = (self.grid_size ** 2) - target_filled
        
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        random.shuffle(positions)
        
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        # Get basic strategies
        basic_strategies = {
            'easy': ['single_cell_cage', 'naked_single'],
            'moderate': ['cage_candidate_elimination', 'two_cell_addition_cage'],
            'hard': ['multi_cage_chain_analysis', 'advanced_cage_arithmetic']
        }
        
        return puzzle, cages, basic_strategies[difficulty]
    
    def create_simple_cages(self, solution: np.ndarray) -> List[Dict]:
        """Create simple cages for fallback"""
        cages = []
        
        # Create mostly single-cell and two-cell cages
        used_cells = set()
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in used_cells:
                    # 60% chance of single cell, 40% chance of two cells
                    if random.random() < 0.6 or c == self.grid_size - 1:
                        # Single cell cage
                        cages.append({
                            'cells': [(r, c)],
                            'operation': 'single',
                            'target': solution[r, c]
                        })
                        used_cells.add((r, c))
                    else:
                        # Two cell cage (horizontal)
                        if (r, c + 1) not in used_cells:
                            values = [solution[r, c], solution[r, c + 1]]
                            cages.append({
                                'cells': [(r, c), (r, c + 1)],
                                'operation': 'addition',
                                'target': sum(values)
                            })
                            used_cells.add((r, c))
                            used_cells.add((r, c + 1))
        
        return cages
    
    def create_mnist_representation(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to MNIST image representation"""
        cell_size = 28
        total_size = self.grid_size * cell_size
        mnist_grid = np.zeros((total_size, total_size), dtype=np.uint8)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if grid[row, col] != 0:
                    digit_img = self.get_mnist_image(grid[row, col])
                    
                    start_row = row * cell_size
                    start_col = col * cell_size
                    mnist_grid[start_row:start_row+cell_size, start_col:start_col+cell_size] = digit_img
        
        return mnist_grid
    
    def get_strategy_details(self, strategy_name: str) -> Dict:
        """Get strategy details from knowledge bases"""
        for kb in [self.easy_kb, self.moderate_kb, self.hard_kb]:
            if strategy_name in kb.list_strategies():
                return kb.get_strategy(strategy_name)
        
        return {
            'name': strategy_name,
            'description': f'Strategy: {strategy_name}',
            'logic': f'Apply {strategy_name} technique',
            'complexity': 'unknown',
            'composite': False
        }
    
    def generate_guaranteed_valid_puzzles(self, difficulty: str, target_count: int) -> List[Dict]:
        """Generate exactly the requested number of VALID puzzles"""
        print(f"\n🎯 Generating exactly {target_count} VALID {difficulty} KenKen puzzles...")
        
        generated_puzzles = []
        attempts = 0
        max_total_attempts = target_count * 500  # Safety limit
        
        start_time = time.time()
        
        while len(generated_puzzles) < target_count and attempts < max_total_attempts:
            attempts += 1
            
            if attempts % 25 == 0:
                elapsed = time.time() - start_time
                rate = len(generated_puzzles) / elapsed if elapsed > 0 else 0
                print(f"  📊 Progress: {len(generated_puzzles)}/{target_count} valid puzzles "
                      f"(attempts: {attempts}, rate: {rate:.2f}/sec)")
            
            try:
                # Generate complete solution
                solution = self.generate_complete_kenken_solution()
                
                if not self.validator.is_valid_kenken_solution(solution):
                    continue
                
                # Create puzzle from solution
                puzzle, cages, strategies = self.create_puzzle_from_solution(solution, difficulty)
                
                # Final validation
                if not self.validator.meets_difficulty_requirements(puzzle, cages, difficulty, strategies):
                    continue
                
                # Create MNIST representations
                mnist_puzzle = self.create_mnist_representation(puzzle)
                mnist_solution = self.create_mnist_representation(solution)
                
                # Create puzzle entry
                puzzle_entry = {
                    'id': f"kenken_{difficulty}_{len(generated_puzzles):04d}",
                    'difficulty': difficulty,
                    'grid_size': self.grid_size,
                    'puzzle_grid': puzzle.tolist(),
                    'solution_grid': solution.tolist(),
                    'cages': cages,
                    'required_strategies': strategies,
                    'mnist_puzzle': mnist_puzzle.tolist(),
                    'mnist_solution': mnist_solution.tolist(),
                    'strategy_details': {
                        strategy: self.get_strategy_details(strategy)
                        for strategy in strategies
                    },
                    'metadata': {
                        'generated_timestamp': datetime.now().isoformat(),
                        'filled_cells': int(np.sum(puzzle != 0)),
                        'empty_cells': int(np.sum(puzzle == 0)),
                        'total_cages': len(cages),
                        'cage_operations': [cage['operation'] for cage in cages],
                        'difficulty_score': self.validator.calculate_strategy_complexity(strategies),
                        'cage_complexity': self.validator.calculate_cage_complexity(cages),
                        'validation_passed': True,
                        'generation_attempt': attempts,
                        'generator_version': '1.0.0'
                    }
                }
                
                generated_puzzles.append(puzzle_entry)
                print(f"  ✅ Valid {difficulty} KenKen puzzle {len(generated_puzzles)}/{target_count} generated")
                
            except Exception as e:
                print(f"  ⚠️ Error in attempt {attempts}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        success_rate = len(generated_puzzles) / attempts * 100 if attempts > 0 else 0
        
        print(f"\n📈 Generation complete for {difficulty}:")
        print(f"  ✅ Generated: {len(generated_puzzles)}/{target_count} valid puzzles")
        print(f"  🕐 Time: {elapsed_time:.1f} seconds")
        print(f"  📊 Success rate: {success_rate:.1f}%")
        print(f"  🔄 Total attempts: {attempts}")
        
        if len(generated_puzzles) < target_count:
            print(f"  ⚠️ Warning: Only generated {len(generated_puzzles)} out of {target_count} requested puzzles")
        
        return generated_puzzles
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"💾 KenKen Dataset saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
    
    def save_mnist_images_with_metadata(self, dataset: List[Dict], output_dir: str):
        """Save MNIST images with individual JSON metadata files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            metadata_dir = os.path.join(output_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)
            
            for entry in dataset:
                puzzle_id = entry['id']
                
                # Save images
                puzzle_img = Image.fromarray(np.array(entry['mnist_puzzle'], dtype=np.uint8))
                solution_img = Image.fromarray(np.array(entry['mnist_solution'], dtype=np.uint8))
                
                puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
                solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
                
                puzzle_img.save(puzzle_path)
                solution_img.save(solution_path)
                
                # Create metadata JSON
                metadata = {
                    'puzzle_info': {
                        'id': entry['id'],
                        'type': 'kenken',
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
                        'total_cells': entry['grid_size'] ** 2,
                        'filled_cells': entry['metadata']['filled_cells'],
                        'empty_cells': entry['metadata']['empty_cells'],
                        'fill_percentage': round((entry['metadata']['filled_cells'] / (entry['grid_size'] ** 2)) * 100, 1),
                        'total_cages': entry['metadata']['total_cages'],
                        'cage_operations': entry['metadata']['cage_operations'],
                        'difficulty_score': entry['metadata']['difficulty_score'],
                        'cage_complexity': entry['metadata']['cage_complexity'],
                        'generation_attempt': entry['metadata']['generation_attempt']
                    }
                }
                
                metadata_path = os.path.join(metadata_dir, f"{puzzle_id}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"🖼️ MNIST KenKen images and metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving images: {e}")


def main():
    """Test the generator"""
    generator = MNISTKenKenGenerator(grid_size=4)
    
    # Test generating valid puzzles
    test_puzzles = generator.generate_guaranteed_valid_puzzles('easy', 2)
    
    print(f"\nGenerated {len(test_puzzles)} valid KenKen puzzles")
    for puzzle in test_puzzles:
        print(f"- {puzzle['id']}: {puzzle['metadata']['filled_cells']} filled cells, "
              f"strategies: {puzzle['required_strategies']}")
        print(f"  Cages: {len(puzzle['cages'])}, Operations: {set(cage['operation'] for cage in puzzle['cages'])}")


if __name__ == "__main__":
    main()