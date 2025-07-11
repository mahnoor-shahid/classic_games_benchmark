# kenken_generator.py
"""
MNIST-based Ken Ken Puzzle Generator with Integrated Validation
Generates exactly the requested number of VALID Ken Ken puzzles
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

# Import knowledge bases and solver
from kenken_easy_strategies_kb import KenKenEasyStrategiesKB
from kenken_moderate_strategies_kb import KenKenModerateStrategiesKB
from kenken_hard_strategies_kb import KenKenHardStrategiesKB
from kenken_solver import KenKenSolver

class KenKenValidator:
    """Integrated Ken Ken validator that ensures puzzle quality"""
    
    def __init__(self):
        self.easy_kb = KenKenEasyStrategiesKB()
        self.moderate_kb = KenKenModerateStrategiesKB()
        self.hard_kb = KenKenHardStrategiesKB()
        self.solver = KenKenSolver()
    
    def is_valid_kenken_solution(self, grid: np.ndarray) -> bool:
        """Check if a complete grid is a valid Ken Ken solution (Latin square)"""
        grid_size = len(grid)
        
        # Check rows
        for row in range(grid_size):
            if set(grid[row, :]) != set(range(1, grid_size + 1)):
                return False
        
        # Check columns
        for col in range(grid_size):
            if set(grid[:, col]) != set(range(1, grid_size + 1)):
                return False
        
        return True
    
    def has_unique_solution(self, puzzle: np.ndarray, cages: List[Dict]) -> bool:
        """Check if puzzle has exactly one solution"""
        solutions_found = 0
        max_solutions = 2
        
        def solve_with_count(grid):
            nonlocal solutions_found
            
            if solutions_found >= max_solutions:
                return
            
            # Find first empty cell
            empty_pos = None
            for row in range(len(grid)):
                for col in range(len(grid)):
                    if grid[row, col] == 0:
                        empty_pos = (row, col)
                        break
                if empty_pos:
                    break
            
            if not empty_pos:
                # Check if this complete grid satisfies all cage constraints
                if self.validate_cage_constraints(grid, cages):
                    solutions_found += 1
                return
            
            row, col = empty_pos
            
            # Try each number
            for num in range(1, len(grid) + 1):
                if self.is_valid_placement(grid, row, col, num, cages):
                    grid[row, col] = num
                    solve_with_count(grid)
                    grid[row, col] = 0
                    
                    if solutions_found >= max_solutions:
                        return
        
        test_grid = puzzle.copy()
        solve_with_count(test_grid)
        return solutions_found == 1
    
    def is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int, cages: List[Dict]) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check Latin square constraints
        if num in grid[row, :] or num in grid[:, col]:
            return False
        
        # Check cage constraints
        cell_cage = self.get_cell_cage(row, col, cages)
        if cell_cage:
            return self.is_cage_placement_valid(grid, row, col, num, cell_cage)
        
        return True
    
    def get_cell_cage(self, row: int, col: int, cages: List[Dict]) -> Optional[Dict]:
        """Find the cage containing the given cell"""
        for cage in cages:
            if (row, col) in cage['cells']:
                return cage
        return None
    
    def is_cage_placement_valid(self, grid: np.ndarray, row: int, col: int, num: int, cage: Dict) -> bool:
        """Check if placing num in cell respects cage constraints"""
        # Temporarily place the number
        grid[row, col] = num
        
        # Get all values in the cage
        cage_values = []
        empty_count = 0
        for r, c in cage['cells']:
            if grid[r, c] != 0:
                cage_values.append(grid[r, c])
            else:
                empty_count += 1
        
        # Restore grid
        grid[row, col] = 0
        
        # If cage is complete, check exact constraint
        if empty_count == 0:
            return self.evaluate_cage_constraint(cage_values, cage['operation'], cage['target'])
        
        # If cage is partial, check if it's still possible to satisfy
        return self.can_cage_be_completed(cage_values, cage['operation'], cage['target'], empty_count, len(grid))
    
    def can_cage_be_completed(self, partial_values: List[int], operation: str, target: int, 
                             empty_count: int, grid_size: int) -> bool:
        """Check if a partially filled cage can still reach the target"""
        if operation == 'add':
            current_sum = sum(partial_values)
            remaining = target - current_sum
            # Check if remaining sum is achievable with empty_count cells
            min_possible = empty_count  # minimum is 1 per cell
            max_possible = empty_count * grid_size  # maximum is grid_size per cell
            return min_possible <= remaining <= max_possible
        
        elif operation == 'multiply':
            current_product = 1
            for val in partial_values:
                current_product *= val
            if target % current_product != 0:
                return False
            remaining_target = target // current_product
            # Check if remaining product is achievable
            return remaining_target >= 1
        
        elif operation in ['subtract', 'divide']:
            # For subtract/divide, we need exactly 2 cells
            return len(partial_values) + empty_count == 2
        
        return True
    
    def validate_cage_constraints(self, grid: np.ndarray, cages: List[Dict]) -> bool:
        """Check if all cage constraints are satisfied"""
        for cage in cages:
            cage_values = [grid[r, c] for r, c in cage['cells']]
            if not self.evaluate_cage_constraint(cage_values, cage['operation'], cage['target']):
                return False
        return True
    
    def evaluate_cage_constraint(self, values: List[int], operation: str, target: int) -> bool:
        """Check if a set of values satisfies a cage constraint"""
        if not values:
            return False
        
        if operation == 'add':
            return sum(values) == target
        
        elif operation == 'subtract':
            if len(values) == 2:
                return abs(values[0] - values[1]) == target
            return False
        
        elif operation == 'multiply':
            result = 1
            for val in values:
                result *= val
            return result == target
        
        elif operation == 'divide':
            if len(values) == 2:
                return max(values) / min(values) == target
            return False
        
        return False
    
    def can_be_solved_with_strategies(self, puzzle: np.ndarray, cages: List[Dict], 
                                    required_strategies: List[str]) -> bool:
        """Check if puzzle can be solved using only the required strategies"""
        try:
            self.solver.grid = puzzle.copy()
            self.solver.grid_size = len(puzzle)
            self.solver.cages = cages
            self.solver.initialize_candidates()
            
            solved_grid, used_strategies = self.solver.solve_puzzle(
                puzzle.copy(), cages, required_strategies, max_time_seconds=30
            )
            
            return self.solver.validate_solution()
        except:
            return False
    
    def meets_difficulty_requirements(self, puzzle: np.ndarray, cages: List[Dict], 
                                     difficulty: str, required_strategies: List[str]) -> bool:
        """Check if puzzle meets all requirements for the difficulty level"""
        if not self.has_unique_solution(puzzle, cages):
            return False
        
        if not self.can_be_solved_with_strategies(puzzle, cages, required_strategies):
            return False
        
        # Additional difficulty-specific checks
        grid_size = len(puzzle)
        filled_cells = np.sum(puzzle != 0)
        
        if difficulty == 'easy':
            return (grid_size <= 5 and 
                    len(cages) <= 8 and
                    all(s in self.easy_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'moderate':
            return (grid_size <= 6 and 
                    len(cages) <= 12 and
                    any(s in self.moderate_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'hard':
            return (grid_size <= 7 and 
                    len(cages) <= 15 and
                    any(s in self.hard_kb.list_strategies() for s in required_strategies))
        
        return False


class MNISTKenKenGenerator:
    """MNIST Ken Ken Generator with guaranteed valid puzzle generation"""
    
    def __init__(self, config_manager=None):
        """Initialize the generator with integrated validation"""
        print("🚀 Initializing MNIST Ken Ken Generator with integrated validation...")
        
        # Initialize validator
        self.validator = KenKenValidator()
        
        # Initialize knowledge bases
        self.easy_kb = KenKenEasyStrategiesKB()
        self.moderate_kb = KenKenModerateStrategiesKB()
        self.hard_kb = KenKenHardStrategiesKB()
        
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
        
        print("✅ Generator initialized successfully")
    
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
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Organize by digit (1-9 only, skip 0)
            train_by_digit = {i: [] for i in range(1, 10)}
            test_by_digit = {i: [] for i in range(1, 10)}
            
            for image, label in train_dataset:
                if 1 <= label <= 9:
                    train_by_digit[label].append(image)
            
            for image, label in test_dataset:
                if 1 <= label <= 9:
                    test_by_digit[label].append(image)
            
            self.mnist_images = {
                'train': train_by_digit,
                'test': test_by_digit
            }
            
            total_images = sum(len(images) for digit_images in train_by_digit.values() for images in [digit_images])
            print(f"✅ MNIST loaded: {total_images} training images for digits 1-9")
            
        except Exception as e:
            print(f"⚠️ Error loading MNIST: {e}")
            print("🔄 Creating fallback dummy data...")
            self.create_dummy_mnist()
    
    def create_dummy_mnist(self):
        """Create dummy MNIST data for testing"""
        print("🎨 Generating dummy MNIST images...")
        
        train_by_digit = {i: [] for i in range(1, 10)}
        test_by_digit = {i: [] for i in range(1, 10)}
        
        for digit in range(1, 10):
            for _ in range(50):  # 50 images per digit
                dummy_img = self.create_digit_pattern(digit)
                train_by_digit[digit].append(dummy_img)
                test_by_digit[digit].append(dummy_img)
        
        self.mnist_images = {'train': train_by_digit, 'test': test_by_digit}
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
        else:
            # Generic pattern with digit-specific characteristics
            for i in range(28):
                for j in range(28):
                    if (i + j + digit * 3) % 8 < 2:
                        img[i, j] = min(255, (i + j + digit * 30) % 256)
        
        return img
    
    def get_mnist_image(self, digit: int) -> np.ndarray:
        """Get a random MNIST image for the digit"""
        if digit < 1 or digit > 9:
            raise ValueError(f"Digit must be 1-9, got {digit}")
        
        available_images = self.mnist_images['train'][digit]
        if not available_images:
            return self.create_digit_pattern(digit)
        
        return random.choice(available_images)
    
    def generate_complete_kenken_solution(self, grid_size: int) -> np.ndarray:
        """Generate a complete valid Ken Ken solution (Latin square)"""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        if self.solve_latin_square(grid, 0, 0):
            return grid
        
        # Fallback: create a simple valid Latin square
        return self.create_simple_latin_square(grid_size)
    
    def solve_latin_square(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Solve Latin square using backtracking"""
        grid_size = len(grid)
        
        if row == grid_size:
            return True
        
        next_row, next_col = (row, col + 1) if col + 1 < grid_size else (row + 1, 0)
        
        numbers = list(range(1, grid_size + 1))
        random.shuffle(numbers)
        
        for num in numbers:
            if self.is_valid_latin_placement(grid, row, col, num):
                grid[row, col] = num
                if self.solve_latin_square(grid, next_row, next_col):
                    return True
                grid[row, col] = 0
        
        return False
    
    def is_valid_latin_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid for Latin square"""
        return num not in grid[row, :] and num not in grid[:, col]
    
    def create_simple_latin_square(self, grid_size: int) -> np.ndarray:
        """Create a simple valid Latin square"""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for row in range(grid_size):
            for col in range(grid_size):
                grid[row, col] = (row + col) % grid_size + 1
        
        return grid
    
    def generate_cages(self, grid: np.ndarray, difficulty: str) -> List[Dict]:
        """Generate cages for the Ken Ken puzzle"""
        grid_size = len(grid)
        complexity = self.get_complexity_settings(difficulty)
        
        num_cages = random.randint(complexity['min_cages'], complexity['max_cages'])
        max_cage_size = complexity['max_cage_size']
        operations = complexity['operations']
        
        cages = []
        used_cells = set()
        
        # Generate cages
        attempts = 0
        while len(cages) < num_cages and len(used_cells) < grid_size * grid_size and attempts < 100:
            attempts += 1
            
            # Choose random starting cell
            available_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) 
                             if (r, c) not in used_cells]
            
            if not available_cells:
                break
            
            start_cell = random.choice(available_cells)
            cage_size = random.randint(1, min(max_cage_size, len(available_cells)))
            
            # Build cage by growing from start cell
            cage_cells = self.build_cage(start_cell, cage_size, used_cells, grid_size)
            
            if cage_cells:
                # Calculate cage target and operation
                cage_values = [grid[r, c] for r, c in cage_cells]
                operation, target = self.calculate_cage_constraint(cage_values, operations)
                
                if operation and target:
                    cage = {
                        'cells': cage_cells,
                        'operation': operation,
                        'target': target
                    }
                    cages.append(cage)
                    used_cells.update(cage_cells)
        
        # Ensure all cells are covered
        uncovered_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) 
                          if (r, c) not in used_cells]
        
        for cell in uncovered_cells:
            # Create single-cell cage
            cage = {
                'cells': [cell],
                'operation': 'add',  # Single cell is just its value
                'target': grid[cell[0], cell[1]]
            }
            cages.append(cage)
            used_cells.add(cell)
        
        return cages
    
    def get_complexity_settings(self, difficulty: str) -> Dict:
        """Get complexity settings for difficulty"""
        settings = {
            'easy': {
                'min_cages': 3,
                'max_cages': 6,
                'max_cage_size': 3,
                'operations': ['add', 'subtract']
            },
            'moderate': {
                'min_cages': 4,
                'max_cages': 9,
                'max_cage_size': 4,
                'operations': ['add', 'subtract', 'multiply']
            },
            'hard': {
                'min_cages': 6,
                'max_cages': 12,
                'max_cage_size': 5,
                'operations': ['add', 'subtract', 'multiply', 'divide']
            }
        }
        return settings.get(difficulty, settings['easy'])
    
    def build_cage(self, start_cell: Tuple[int, int], target_size: int, 
                   used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build a connected cage starting from start_cell"""
        cage_cells = [start_cell]
        candidates = self.get_adjacent_cells(start_cell, used_cells, grid_size)
        
        while len(cage_cells) < target_size and candidates:
            next_cell = random.choice(list(candidates))
            cage_cells.append(next_cell)
            
            # Update candidates with neighbors of new cell
            new_candidates = self.get_adjacent_cells(next_cell, used_cells | set(cage_cells), grid_size)
            candidates.update(new_candidates)
            candidates.discard(next_cell)
        
        return cage_cells
    
    def get_adjacent_cells(self, cell: Tuple[int, int], used_cells: Set[Tuple[int, int]], 
                          grid_size: int) -> Set[Tuple[int, int]]:
        """Get adjacent cells that are not used"""
        row, col = cell
        adjacent = set()
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < grid_size and 0 <= new_col < grid_size and 
                (new_row, new_col) not in used_cells):
                adjacent.add((new_row, new_col))
        
        return adjacent
    
    def calculate_cage_constraint(self, values: List[int], allowed_operations: List[str]) -> Tuple[str, int]:
        """Calculate operation and target for cage values"""
        if len(values) == 1:
            return 'add', values[0]
        
        # Try different operations
        if 'add' in allowed_operations:
            return 'add', sum(values)
        
        if 'multiply' in allowed_operations and len(values) <= 3:
            product = 1
            for val in values:
                product *= val
            if product <= 100:  # Reasonable target
                return 'multiply', product
        
        if len(values) == 2:
            if 'subtract' in allowed_operations:
                return 'subtract', abs(values[0] - values[1])
            
            if 'divide' in allowed_operations:
                max_val, min_val = max(values), min(values)
                if min_val != 0 and max_val % min_val == 0:
                    return 'divide', max_val // min_val
        
        # Fallback to addition
        return 'add', sum(values)
    
    def create_puzzle_from_solution(self, solution: np.ndarray, cages: List[Dict], difficulty: str) -> np.ndarray:
        """Create puzzle by removing some digits strategically"""
        grid_size = len(solution)
        puzzle = solution.copy()
        
        # For Ken Ken, we typically start with fewer filled cells than Sudoku
        if difficulty == 'easy':
            cells_to_clear = random.randint(int(grid_size * grid_size * 0.4), int(grid_size * grid_size * 0.6))
        elif difficulty == 'moderate':
            cells_to_clear = random.randint(int(grid_size * grid_size * 0.5), int(grid_size * grid_size * 0.7))
        else:  # hard
            cells_to_clear = random.randint(int(grid_size * grid_size * 0.6), int(grid_size * grid_size * 0.8))
        
        # Get all cell positions
        all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(all_positions)
        
        cleared = 0
        for row, col in all_positions:
            if cleared >= cells_to_clear:
                break
            
            # Try clearing this cell
            original_value = puzzle[row, col]
            puzzle[row, col] = 0
            
            # Check if puzzle still has unique solution
            if self.validator.has_unique_solution(puzzle, cages):
                cleared += 1
            else:
                # Restore if clearing breaks uniqueness
                puzzle[row, col] = original_value
        
        return puzzle
    
    def create_mnist_representation(self, grid: np.ndarray, cages: List[Dict]) -> np.ndarray:
        """Convert grid to MNIST image representation with enhanced cage boundaries"""
        grid_size = len(grid)
        cell_size = 64  # Larger cells for better visibility
        total_size = grid_size * cell_size
        
        mnist_grid = np.zeros((total_size, total_size), dtype=np.uint8)
        
        # Place MNIST digits with padding for boundaries
        for row in range(grid_size):
            for col in range(grid_size):
                if grid[row, col] != 0:
                    digit_img = self.get_mnist_image(grid[row, col])
                    
                    # Resize MNIST image to fit cell with boundary padding
                    from PIL import Image
                    pil_img = Image.fromarray(digit_img).resize((cell_size - 8, cell_size - 8))
                    digit_img_resized = np.array(pil_img, dtype=np.uint8)
                    
                    start_row = row * cell_size + 4
                    start_col = col * cell_size + 4
                    end_row = start_row + cell_size - 8
                    end_col = start_col + cell_size - 8
                    
                    mnist_grid[start_row:end_row, start_col:end_col] = digit_img_resized
        
        # Add enhanced cage boundaries
        self._draw_cage_boundaries(mnist_grid, cages, cell_size, grid_size)
        
        # Add operation labels
        self._add_cage_operation_labels(mnist_grid, cages, cell_size)
        
        return mnist_grid
    
    def _draw_cage_boundaries(self, mnist_grid: np.ndarray, cages: List[Dict], cell_size: int, grid_size: int):
        """Draw enhanced cage boundaries with better visibility"""
        boundary_thickness = 3
        corner_size = 8
        
        for cage_idx, cage in enumerate(cages):
            cage_color = 100 + (cage_idx * 25) % 155  # Different shades
            
            for r, c in cage['cells']:
                pixel_row = r * cell_size
                pixel_col = c * cell_size
                
                # Check which sides need boundaries
                needs_top = (r - 1, c) not in cage['cells']
                needs_bottom = (r + 1, c) not in cage['cells']
                needs_left = (r, c - 1) not in cage['cells']
                needs_right = (r, c + 1) not in cage['cells']
                
                # Draw thick boundaries
                if needs_top:
                    mnist_grid[pixel_row:pixel_row + boundary_thickness, 
                              pixel_col:pixel_col + cell_size] = cage_color
                if needs_bottom:
                    mnist_grid[pixel_row + cell_size - boundary_thickness:pixel_row + cell_size, 
                              pixel_col:pixel_col + cell_size] = cage_color
                if needs_left:
                    mnist_grid[pixel_row:pixel_row + cell_size, 
                              pixel_col:pixel_col + boundary_thickness] = cage_color
                if needs_right:
                    mnist_grid[pixel_row:pixel_row + cell_size, 
                              pixel_col + cell_size - boundary_thickness:pixel_col + cell_size] = cage_color
                
                # Add bright corner markers
                if needs_top and needs_left:
                    mnist_grid[pixel_row:pixel_row + corner_size, 
                              pixel_col:pixel_col + corner_size] = 255
                if needs_top and needs_right:
                    mnist_grid[pixel_row:pixel_row + corner_size, 
                              pixel_col + cell_size - corner_size:pixel_col + cell_size] = 255
                if needs_bottom and needs_left:
                    mnist_grid[pixel_row + cell_size - corner_size:pixel_row + cell_size, 
                              pixel_col:pixel_col + corner_size] = 255
                if needs_bottom and needs_right:
                    mnist_grid[pixel_row + cell_size - corner_size:pixel_row + cell_size, 
                              pixel_col + cell_size - corner_size:pixel_col + cell_size] = 255
    
    def _add_cage_operation_labels(self, mnist_grid: np.ndarray, cages: List[Dict], cell_size: int):
        """Add operation and target labels to cages"""
        for cage in cages:
            if len(cage['cells']) == 1:
                continue
            
            # Get top-left cell
            min_row = min(r for r, c in cage['cells'])
            min_col = min(c for r, c in cage['cells'])
            
            label_row = min_row * cell_size + 2
            label_col = min_col * cell_size + 2
            
            # Add operation symbol and target (simplified as bright pixels)
            operation_color = 255
            
            # Draw operation indicator
            if cage['operation'] == 'add':
                # Plus sign
                mnist_grid[label_row + 4:label_row + 8, label_col + 2:label_col + 10] = operation_color
                mnist_grid[label_row + 2:label_row + 10, label_col + 4:label_col + 8] = operation_color
            elif cage['operation'] == 'subtract':
                # Minus sign
                mnist_grid[label_row + 4:label_row + 8, label_col + 2:label_col + 10] = operation_color
            elif cage['operation'] == 'multiply':
                # X shape
                for i in range(8):
                    if label_row + 2 + i < mnist_grid.shape[0] and label_col + 2 + i < mnist_grid.shape[1]:
                        mnist_grid[label_row + 2 + i, label_col + 2 + i] = operation_color
                    if label_row + 2 + i < mnist_grid.shape[0] and label_col + 10 - i >= 0:
                        mnist_grid[label_row + 2 + i, label_col + 10 - i] = operation_color
            elif cage['operation'] == 'divide':
                # Division symbol
                mnist_grid[label_row + 2:label_row + 4, label_col + 4:label_col + 8] = operation_color
                mnist_grid[label_row + 6:label_row + 8, label_col + 2:label_col + 10] = operation_color
                mnist_grid[label_row + 10:label_row + 12, label_col + 4:label_col + 8] = operation_color
    
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
    
    def get_random_strategies(self, difficulty: str) -> List[str]:
        """Get appropriate random strategies for difficulty"""
        if difficulty == 'easy':
            available = list(self.easy_kb.list_strategies())
            return random.sample(available, min(3, len(available)))
        
        elif difficulty == 'moderate':
            easy_strategies = list(self.easy_kb.list_strategies())
            moderate_strategies = list(self.moderate_kb.list_strategies())
            
            # Must include at least one moderate strategy
            selected = [random.choice(moderate_strategies)]
            remaining = easy_strategies + moderate_strategies
            remaining = [s for s in remaining if s not in selected]
            
            num_additional = random.randint(2, 4)
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
            
            num_additional = random.randint(3, 6)
            selected.extend(random.sample(remaining, min(num_additional, len(remaining))))
            return selected
    
    def generate_guaranteed_valid_puzzles(self, difficulty: str, target_count: int, grid_size: int = None) -> List[Dict]:
        """Generate exactly the requested number of VALID Ken Ken puzzles"""
        print(f"\n🎯 Generating exactly {target_count} VALID {difficulty} Ken Ken puzzles...")
        
        if grid_size is None:
            grid_sizes = self.get_grid_sizes_for_difficulty(difficulty)
            if difficulty == 'easy':
                grid_size = random.choice(grid_sizes)  # Randomly pick 4 or 5 for easy
            else:
                grid_size = grid_sizes  # Single value for moderate/hard
        
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
                solution = self.generate_complete_kenken_solution(grid_size)
                
                if not self.validator.is_valid_kenken_solution(solution):
                    continue
                
                # Generate cages
                cages = self.generate_cages(solution, difficulty)
                
                # Create puzzle from solution
                puzzle = self.create_puzzle_from_solution(solution, cages, difficulty)
                
                # Get strategies for this difficulty
                strategies = self.get_random_strategies(difficulty)
                
                # Final validation
                if not self.validator.meets_difficulty_requirements(puzzle, cages, difficulty, strategies):
                    continue
                
                # Create MNIST representations
                mnist_puzzle = self.create_mnist_representation(puzzle, cages)
                mnist_solution = self.create_mnist_representation(solution, cages)
                
                # Create puzzle entry
                puzzle_entry = {
                    'id': f"{difficulty}_{grid_size}x{grid_size}_{len(generated_puzzles):04d}",
                    'difficulty': difficulty,
                    'grid_size': grid_size,
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
                        'num_cages': len(cages),
                        'operations_used': list(set(cage['operation'] for cage in cages)),
                        'difficulty_score': self.calculate_difficulty_score(strategies, cages),
                        'validation_passed': True,
                        'generation_attempt': attempts,
                        'generator_version': '1.0.0'
                    }
                }
                
                generated_puzzles.append(puzzle_entry)
                print(f"  ✅ Valid {difficulty} {grid_size}x{grid_size} puzzle {len(generated_puzzles)}/{target_count} generated")
                
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
    
    def get_grid_sizes_for_difficulty(self, difficulty: str):
        """Get appropriate grid size(s) for difficulty"""
        if difficulty == 'easy':
            return [4, 5]  # Mix of 4x4 and 5x5
        elif difficulty == 'moderate':
            return 6  # Only 6x6
        else:  # hard
            return 7  # Only 7x7
    
    def calculate_difficulty_score(self, strategies: List[str], cages: List[Dict]) -> float:
        """Calculate difficulty score based on strategies and cages"""
        strategy_complexity = {
            # Easy strategies
            'single_cell_cage': 0.5,
            'simple_addition_cage': 1.0,
            'simple_subtraction_cage': 1.2,
            'naked_single': 0.8,
            'cage_completion': 1.0,
            
            # Moderate strategies
            'cage_elimination': 2.0,
            'complex_arithmetic_cages': 2.5,
            'constraint_propagation': 2.2,
            'multi_cage_analysis': 3.0,
            
            # Hard strategies
            'advanced_cage_chaining': 4.0,
            'constraint_satisfaction_solving': 4.5,
        }
        
        base_score = sum(strategy_complexity.get(strategy, 1.0) for strategy in strategies)
        cage_bonus = len(cages) * 0.1
        operation_bonus = len(set(cage['operation'] for cage in cages)) * 0.2
        
        return base_score + cage_bonus + operation_bonus
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2, cls=NumpyEncoder)
            print(f"💾 Dataset saved to {filename}")
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
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"🖼️ MNIST images and metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving images: {e}")


def main():
    """Test the generator"""
    generator = MNISTKenKenGenerator()
    
    # Test generating valid puzzles
    test_puzzles = generator.generate_guaranteed_valid_puzzles('easy', 2, 4)
    
    print(f"\nGenerated {len(test_puzzles)} valid Ken Ken puzzles")
    for puzzle in test_puzzles:
        print(f"- {puzzle['id']}: {puzzle['grid_size']}x{puzzle['grid_size']} grid, "
              f"{puzzle['metadata']['filled_cells']} filled cells, "
              f"{puzzle['metadata']['num_cages']} cages, "
              f"strategies: {puzzle['required_strategies']}")


if __name__ == "__main__":
    main()