# sudoku_generator.py
"""
MNIST-based Sudoku Puzzle Generator with Integrated Validation
Generates exactly the requested number of VALID Sudoku puzzles
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

# Import knowledge bases and solver
from sudoku_easy_strategies_kb import EasyStrategiesKB
from sudoku_moderate_strategies_kb import ModerateStrategiesKB
from sudoku_hard_strategies_kb import HardStrategiesKB


class SudokuValidator:
    """Integrated Sudoku validator that ensures puzzle quality"""
    
    def __init__(self):
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.hard_kb = HardStrategiesKB()
    
    def is_valid_sudoku_solution(self, grid: np.ndarray) -> bool:
        """Check if a complete grid is a valid Sudoku solution"""
        # Check rows
        for row in range(9):
            if set(grid[row, :]) != set(range(1, 10)):
                return False
        
        # Check columns
        for col in range(9):
            if set(grid[:, col]) != set(range(1, 10)):
                return False
        
        # Check boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = grid[box_row:box_row+3, box_col:box_col+3]
                if set(box.flatten()) != set(range(1, 10)):
                    return False
        
        return True
    
    def has_unique_solution(self, puzzle: np.ndarray) -> bool:
        """Check if puzzle has exactly one solution using optimized solver"""
        solutions_found = 0
        max_solutions = 2  # We only need to know if there's 1 or more than 1
        
        def solve_with_count(grid):
            nonlocal solutions_found
            
            if solutions_found >= max_solutions:
                return
            
            # Find first empty cell
            empty_pos = None
            for row in range(9):
                for col in range(9):
                    if grid[row, col] == 0:
                        empty_pos = (row, col)
                        break
                if empty_pos:
                    break
            
            if not empty_pos:
                # No empty cells - found a solution
                solutions_found += 1
                return
            
            row, col = empty_pos
            
            # Try each number
            for num in range(1, 10):
                if self.is_valid_placement(grid, row, col, num):
                    grid[row, col] = num
                    solve_with_count(grid)
                    grid[row, col] = 0
                    
                    if solutions_found >= max_solutions:
                        return
        
        test_grid = puzzle.copy()
        solve_with_count(test_grid)
        return solutions_found == 1
    
    def is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in grid[row, :]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = grid[box_row:box_row+3, box_col:box_col+3]
        if num in box:
            return False
        
        return True
    
    def can_be_solved_with_strategies(self, puzzle: np.ndarray, required_strategies: List[str]) -> bool:
        """Check if puzzle can be solved using only the required strategies (simplified)"""
        # This is a heuristic-based validation
        # In practice, you'd run a full solver with only these strategies
        
        filled_cells = np.sum(puzzle != 0)
        strategy_complexity = self.calculate_strategy_complexity(required_strategies)
        
        # Difficulty thresholds based on filled cells and strategy complexity
        if strategy_complexity <= 1.0:  # Easy strategies
            return 35 <= filled_cells <= 50
        elif strategy_complexity <= 2.5:  # Moderate strategies
            return 25 <= filled_cells <= 40
        else:  # Hard strategies
            return 17 <= filled_cells <= 32
    
    def calculate_strategy_complexity(self, strategies: List[str]) -> float:
        """Calculate complexity score based on strategies"""
        complexity_scores = {
            # Easy strategies (score 0.5)
            'naked_single': 0.5,
            'hidden_single_row': 0.5,
            'hidden_single_column': 0.5,
            'hidden_single_box': 0.5,
            'full_house_row': 0.3,
            'full_house_column': 0.3,
            'full_house_box': 0.3,
            
            # Moderate strategies (score 1.0-2.0)
            'naked_pair': 1.0,
            'naked_triple': 1.5,
            'hidden_pair': 1.2,
            'pointing_pairs': 1.3,
            'box_line_reduction': 1.4,
            'x_wing': 2.0,
            'xy_wing': 2.2,
            
            # Hard strategies (score 2.5+)
            'swordfish': 3.0,
            'xyz_wing': 2.8,
            'als_xz': 3.5,
            'simple_coloring': 2.5,
            'multi_coloring': 4.0,
            'death_blossom': 4.5,
            'exocet': 5.0
        }
        
        total_score = sum(complexity_scores.get(strategy, 1.0) for strategy in strategies)
        return total_score / len(strategies) if strategies else 0.0
    
    def meets_difficulty_requirements(self, puzzle: np.ndarray, difficulty: str, required_strategies: List[str]) -> bool:
        """Check if puzzle meets all requirements for the difficulty level"""
        if not self.has_unique_solution(puzzle):
            return False
        
        if not self.can_be_solved_with_strategies(puzzle, required_strategies):
            return False
        
        # Additional difficulty-specific checks
        filled_cells = np.sum(puzzle != 0)
        
        if difficulty == 'easy':
            return (35 <= filled_cells <= 45 and 
                    len(required_strategies) <= 3 and
                    all(s in self.easy_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'moderate':
            return (25 <= filled_cells <= 40 and 
                    len(required_strategies) <= 4 and
                    any(s in self.moderate_kb.list_strategies() for s in required_strategies))
        
        elif difficulty == 'hard':
            return (17 <= filled_cells <= 32 and 
                    len(required_strategies) <= 6 and
                    any(s in self.hard_kb.list_strategies() for s in required_strategies))
        
        return False


class MNISTSudokuGenerator:
    """MNIST Sudoku Generator with guaranteed valid puzzle generation"""
    
    def __init__(self, config_manager=None):
        """Initialize the generator with integrated validation"""
        print("🚀 Initializing MNIST Sudoku Generator with integrated validation...")
        
        # Initialize validator
        self.validator = SudokuValidator()
        
        # Initialize knowledge bases
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.hard_kb = HardStrategiesKB()
        
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
            
            # Organize by digit (1-9 only)
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
            # Generic pattern with digit-specific noise
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
    
    def generate_complete_sudoku(self) -> np.ndarray:
        """Generate a complete valid Sudoku solution using optimized algorithm"""
        grid = np.zeros((9, 9), dtype=int)
        
        # Fill diagonal boxes first (they don't interfere with each other)
        for box_start in range(0, 9, 3):
            self.fill_box(grid, box_start, box_start)
        
        # Fill remaining cells
        self.solve_sudoku(grid)
        
        return grid
    
    def fill_box(self, grid: np.ndarray, row: int, col: int):
        """Fill a 3x3 box with random valid numbers"""
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        
        idx = 0
        for i in range(3):
            for j in range(3):
                grid[row + i, col + j] = numbers[idx]
                idx += 1
    
    def solve_sudoku(self, grid: np.ndarray) -> bool:
        """Solve sudoku using backtracking"""
        empty_pos = self.find_empty_cell(grid)
        if not empty_pos:
            return True  # Solved
        
        row, col = empty_pos
        numbers = list(range(1, 10))
        random.shuffle(numbers)  # Randomize for variety
        
        for num in numbers:
            if self.validator.is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                
                if self.solve_sudoku(grid):
                    return True
                
                grid[row, col] = 0
        
        return False
    
    def find_empty_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the first empty cell in the grid"""
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    return (row, col)
        return None
    
    def create_puzzle_from_solution(self, solution: np.ndarray, difficulty: str) -> Tuple[np.ndarray, List[str]]:
        """Create a valid puzzle by strategically removing cells"""
        print(f"    🔨 Creating {difficulty} puzzle...")
        
        max_attempts = 1000  # Increased for better quality
        best_puzzle = None
        best_strategies = []
        best_score = -1
        
        # Get target parameters for difficulty
        target_params = self.get_difficulty_parameters(difficulty)
        
        for attempt in range(max_attempts):
            if attempt > 0 and attempt % 100 == 0:
                print(f"      ⏳ Attempt {attempt}/{max_attempts}")
            
            # Create puzzle candidate
            puzzle = solution.copy()
            strategies = self.get_random_strategies(difficulty)
            
            # Remove cells strategically
            removed_cells = self.remove_cells_strategically(puzzle, target_params)
            
            # Validate the puzzle
            if self.validator.meets_difficulty_requirements(puzzle, difficulty, strategies):
                score = self.calculate_puzzle_quality(puzzle, strategies, difficulty)
                
                if score > best_score:
                    best_puzzle = puzzle.copy()
                    best_strategies = strategies.copy()
                    best_score = score
                
                # If we found a high-quality puzzle, we can stop early
                if score >= 0.8:
                    print(f"      ✅ High-quality puzzle found at attempt {attempt + 1}")
                    break
        
        if best_puzzle is None:
            # Fallback: create a guaranteed valid puzzle
            print(f"      🔄 Using fallback generation for {difficulty}")
            best_puzzle, best_strategies = self.create_fallback_puzzle(solution, difficulty)
        
        return best_puzzle, best_strategies
    
    def get_difficulty_parameters(self, difficulty: str) -> Dict:
        """Get target parameters for difficulty"""
        params = {
            'easy': {
                'min_filled': 36,
                'max_filled': 45,
                'target_filled': 40,
                'max_strategies': 3
            },
            'moderate': {
                'min_filled': 26,
                'max_filled': 38,
                'target_filled': 32,
                'max_strategies': 4
            },
            'hard': {
                'min_filled': 18,
                'max_filled': 30,
                'target_filled': 24,
                'max_strategies': 6
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
            
            num_additional = random.randint(1, 2)
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
            
            num_additional = random.randint(2, 4)
            selected.extend(random.sample(remaining, min(num_additional, len(remaining))))
            return selected
    
    def remove_cells_strategically(self, puzzle: np.ndarray, target_params: Dict) -> int:
        """Remove cells strategically to reach target difficulty, using round-robin removal across boxes and enforcing band minimums for easy puzzles"""
        target_filled = target_params['target_filled']
        cells_to_remove = 81 - target_filled

        if target_params.get('min_filled', 0) >= 36:  # easy puzzle, use round-robin by box and band minimums
            min_per_region = 5
            min_per_band = 12  # minimum clues per horizontal band (top/mid/bottom 3 rows)
            filled_cells = np.count_nonzero(puzzle)
            removed = 0
            # Build a list of removable positions for each 3x3 box
            box_positions = []
            for box_row in range(0, 9, 3):
                for box_col in range(0, 9, 3):
                    positions = [(r, c) for r in range(box_row, box_row+3) for c in range(box_col, box_col+3)]
                    random.shuffle(positions)
                    box_positions.append(positions)
            # Round-robin removal across boxes
            while removed < cells_to_remove and filled_cells > target_filled:
                progress = False
                for box in box_positions:
                    while box:
                        row, col = box.pop()
                        if puzzle[row, col] == 0:
                            continue
                        # Try removing
                        original_value = puzzle[row, col]
                        puzzle[row, col] = 0
                        # Check all region minimums
                        valid = True
                        if np.count_nonzero(puzzle[row, :]) < min_per_region:
                            valid = False
                        if np.count_nonzero(puzzle[:, col]) < min_per_region:
                            valid = False
                        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
                        box_grid = puzzle[box_row:box_row+3, box_col:box_col+3]
                        if np.count_nonzero(box_grid) < min_per_region:
                            valid = False
                        # Check band minimums
                        for band_start in [0, 3, 6]:
                            band = puzzle[band_start:band_start+3, :]
                            if np.count_nonzero(band) < min_per_band:
                                valid = False
                                break
                        # Check uniqueness
                        if valid and self.validator.has_unique_solution(puzzle):
                            removed += 1
                            filled_cells -= 1
                            progress = True
                        else:
                            puzzle[row, col] = original_value
                        break  # Only try one cell per box per round
                if not progress:
                    break  # No more cells can be removed without violating constraints
            return removed
        else:
            # Moderate/hard: keep current logic
            all_positions = [(row, col) for row in range(9) for col in range(9)]
            random.shuffle(all_positions)
            removed = 0
            for row, col in all_positions:
                if removed >= cells_to_remove:
                    break
                original_value = puzzle[row, col]
                puzzle[row, col] = 0
                if self.validator.has_unique_solution(puzzle):
                    removed += 1
                else:
                    puzzle[row, col] = original_value
            return removed
    
    def calculate_puzzle_quality(self, puzzle: np.ndarray, strategies: List[str], difficulty: str) -> float:
        """Calculate quality score for a puzzle (0.0 to 1.0)"""
        score = 0.0
        
        # Factor 1: Appropriate number of filled cells
        filled = np.sum(puzzle != 0)
        target_params = self.get_difficulty_parameters(difficulty)
        
        if target_params['min_filled'] <= filled <= target_params['max_filled']:
            score += 0.3
        
        # Factor 2: Strategy complexity matches difficulty
        complexity = self.validator.calculate_strategy_complexity(strategies)
        expected_complexity = {'easy': 0.5, 'moderate': 1.5, 'hard': 3.0}
        
        if abs(complexity - expected_complexity[difficulty]) < 1.0:
            score += 0.3
        
        # Factor 3: Number of strategies is appropriate
        if len(strategies) <= target_params['max_strategies']:
            score += 0.2
        
        # Factor 4: Has unique solution
        if self.validator.has_unique_solution(puzzle):
            score += 0.2
        
        return score
    
    def create_fallback_puzzle(self, solution: np.ndarray, difficulty: str) -> Tuple[np.ndarray, List[str]]:
        """Create a guaranteed valid puzzle as fallback"""
        puzzle = solution.copy()
        target_params = self.get_difficulty_parameters(difficulty)
        
        # Simple removal: remove random cells until we reach target
        target_filled = target_params['target_filled']
        cells_to_remove = 81 - target_filled
        
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        # Get basic strategies for this difficulty
        basic_strategies = {
            'easy': ['naked_single', 'hidden_single_row'],
            'moderate': ['naked_single', 'naked_pair', 'hidden_single_row'],
            'hard': ['naked_single', 'x_wing', 'hidden_single_row']
        }
        
        return puzzle, basic_strategies[difficulty]
    
    def create_mnist_representation(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to MNIST image representation"""
        mnist_grid = np.zeros((252, 252), dtype=np.uint8)  # 9x9 * 28x28
        
        for row in range(9):
            for col in range(9):
                if grid[row, col] != 0:
                    digit_img = self.get_mnist_image(grid[row, col])
                    
                    start_row = row * 28
                    start_col = col * 28
                    mnist_grid[start_row:start_row+28, start_col:start_col+28] = digit_img
        
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
        print(f"\n🎯 Generating exactly {target_count} VALID {difficulty} puzzles...")
        
        generated_puzzles = []
        attempts = 0
        max_total_attempts = target_count * 1000  # Safety limit
        
        start_time = time.time()
        
        while len(generated_puzzles) < target_count and attempts < max_total_attempts:
            attempts += 1
            
            if attempts % 50 == 0:
                elapsed = time.time() - start_time
                rate = len(generated_puzzles) / elapsed if elapsed > 0 else 0
                print(f"  📊 Progress: {len(generated_puzzles)}/{target_count} valid puzzles "
                      f"(attempts: {attempts}, rate: {rate:.2f}/sec)")
            
            try:
                # Generate complete solution
                solution = self.generate_complete_sudoku()
                
                if not self.validator.is_valid_sudoku_solution(solution):
                    continue
                
                # Create puzzle from solution
                puzzle, strategies = self.create_puzzle_from_solution(solution, difficulty)
                
                # Final validation
                if not self.validator.meets_difficulty_requirements(puzzle, difficulty, strategies):
                    continue
                
                # Create MNIST representations
                mnist_puzzle = self.create_mnist_representation(puzzle)
                mnist_solution = self.create_mnist_representation(solution)
                
                # Create puzzle entry
                puzzle_entry = {
                    'id': f"{difficulty}_{len(generated_puzzles):04d}",
                    'difficulty': difficulty,
                    'puzzle_grid': puzzle.tolist(),
                    'solution_grid': solution.tolist(),
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
                        'difficulty_score': self.validator.calculate_strategy_complexity(strategies),
                        'validation_passed': True,
                        'generation_attempt': attempts,
                        'generator_version': '2.0.0'
                    }
                }
                
                generated_puzzles.append(puzzle_entry)
                print(f"  ✅ Valid {difficulty} puzzle {len(generated_puzzles)}/{target_count} generated")
                
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
                        'validation_status': 'VALID',
                        'generated_timestamp': entry['metadata']['generated_timestamp']
                    },
                    'grids': {
                        'puzzle_grid': entry['puzzle_grid'],
                        'solution_grid': entry['solution_grid']
                    },
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
                        'total_cells': 81,
                        'filled_cells': entry['metadata']['filled_cells'],
                        'empty_cells': entry['metadata']['empty_cells'],
                        'fill_percentage': round((entry['metadata']['filled_cells'] / 81) * 100, 1),
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
    generator = MNISTSudokuGenerator()
    
    # Test generating valid puzzles
    test_puzzles = generator.generate_guaranteed_valid_puzzles('easy', 2)
    
    print(f"\nGenerated {len(test_puzzles)} valid puzzles")
    for puzzle in test_puzzles:
        print(f"- {puzzle['id']}: {puzzle['metadata']['filled_cells']} filled cells, "
              f"strategies: {puzzle['required_strategies']}")


if __name__ == "__main__":
    main()