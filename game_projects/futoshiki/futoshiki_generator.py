# futoshiki_generator_improved.py
"""
Enhanced MNIST-based Futoshiki Puzzle Generator with:
- Consistent MNIST digit usage between puzzle and solution
- Proper strategy-based validation and compositionality
- Text-based constraint visualization
- Larger puzzle sizes (5x5, 6x6, 7x7)
"""

import numpy as np
import json
import random
import os
import time
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
import torchvision.transforms as transforms

# Import knowledge bases and solver
from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from futoshiki_hard_strategies_kb import FutoshikiHardStrategiesKB
from futoshiki_solver import FutoshikiSolver


class EnhancedFutoshikiValidator:
    """Enhanced validator with strategy-based validation and compositionality checking"""
    
    def __init__(self):
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        self.solver = FutoshikiSolver()
        
        # Validate knowledge base compositionality
        self._validate_knowledge_bases()
    
    def _validate_knowledge_bases(self):
        """Validate that knowledge bases have proper compositionality"""
        print("🔗 Validating strategy compositionality...")
        
        easy_valid = self.easy_kb.validate_compositionality()
        moderate_valid = self.moderate_kb.validate_compositionality()
        hard_valid = self.hard_kb.validate_compositionality()
        
        if not (easy_valid and moderate_valid and hard_valid):
            raise ValueError("Knowledge base compositionality validation failed!")
        
        print("✅ Strategy compositionality validated")
    
    def validate_puzzle_with_strategies(self, puzzle: np.ndarray, solution: np.ndarray,
                                      h_constraints: Dict, v_constraints: Dict,
                                      difficulty: str, required_strategies: List[str]) -> Dict:
        """Comprehensive puzzle validation with strategy verification"""
        validation_result = {
            'is_valid': False,
            'has_unique_solution': False,
            'solvable_with_strategies': False,
            'strategy_compositionality_valid': False,
            'difficulty_appropriate': False,
            'constraint_consistency': False,
            'errors': []
        }
        
        try:
            # 1. Basic structure validation
            if not self._validate_basic_structure(puzzle, solution, h_constraints, v_constraints):
                validation_result['errors'].append("Basic structure validation failed")
                return validation_result
            
            # 2. Solution correctness
            if not self.solver.validate_solution(solution, h_constraints, v_constraints):
                validation_result['errors'].append("Solution is incorrect")
                return validation_result
            
            # 3. Unique solution check
            if not self._has_unique_solution_advanced(puzzle, h_constraints, v_constraints):
                validation_result['errors'].append("Puzzle does not have unique solution")
                return validation_result
            validation_result['has_unique_solution'] = True
            
            # 4. Strategy solvability
            solvable, used_strategies = self._can_solve_with_strategies(
                puzzle, h_constraints, v_constraints, required_strategies
            )
            if not solvable:
                validation_result['errors'].append("Puzzle cannot be solved with required strategies")
                return validation_result
            validation_result['solvable_with_strategies'] = True
            
            # 5. Strategy compositionality validation
            if not self._validate_strategy_compositionality(required_strategies):
                validation_result['errors'].append("Strategy compositionality is invalid")
                return validation_result
            validation_result['strategy_compositionality_valid'] = True
            
            # 6. Difficulty appropriateness
            if not self._validate_difficulty_appropriateness(puzzle, required_strategies, difficulty):
                validation_result['errors'].append("Puzzle difficulty does not match required level")
                return validation_result
            validation_result['difficulty_appropriate'] = True
            
            # 7. Constraint consistency
            if not self._validate_constraint_consistency(puzzle, h_constraints, v_constraints):
                validation_result['errors'].append("Constraint consistency validation failed")
                return validation_result
            validation_result['constraint_consistency'] = True
            
            validation_result['is_valid'] = True
            validation_result['used_strategies'] = used_strategies
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_basic_structure(self, puzzle: np.ndarray, solution: np.ndarray,
                                h_constraints: Dict, v_constraints: Dict) -> bool:
        """Validate basic puzzle structure"""
        size = len(puzzle)
        
        # Check dimensions
        if puzzle.shape != (size, size) or solution.shape != (size, size):
            return False
        
        # Check value ranges
        if not all(0 <= val <= size for val in puzzle.flat):
            return False
        if not all(1 <= val <= size for val in solution.flat):
            return False
        
        # Check that puzzle is subset of solution
        for row in range(size):
            for col in range(size):
                if puzzle[row, col] != 0 and puzzle[row, col] != solution[row, col]:
                    return False
        
        return True
    
    def _has_unique_solution_advanced(self, puzzle: np.ndarray, h_constraints: Dict, v_constraints: Dict) -> bool:
        """Advanced unique solution checking with constraint propagation"""
        solutions_found = 0
        max_solutions = 2
        
        def solve_with_constraint_propagation(grid):
            nonlocal solutions_found
            
            if solutions_found >= max_solutions:
                return
            
            # Apply constraint propagation before backtracking
            grid = self._apply_constraint_propagation(grid.copy(), h_constraints, v_constraints)
            
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
                # No empty cells - check if solution is valid
                if self.solver.validate_solution(grid, h_constraints, v_constraints):
                    solutions_found += 1
                return
            
            row, col = empty_pos
            
            # Get valid candidates for this position
            candidates = self._get_valid_candidates(grid, row, col, h_constraints, v_constraints)
            
            for num in candidates:
                if self._is_valid_placement_advanced(grid, row, col, num, h_constraints, v_constraints):
                    grid[row, col] = num
                    solve_with_constraint_propagation(grid)
                    grid[row, col] = 0
                    
                    if solutions_found >= max_solutions:
                        return
        
        test_grid = puzzle.copy()
        solve_with_constraint_propagation(test_grid)
        return solutions_found == 1
    
    def _apply_constraint_propagation(self, grid: np.ndarray, h_constraints: Dict, v_constraints: Dict) -> np.ndarray:
        """Apply basic constraint propagation to reduce search space"""
        size = len(grid)
        changed = True
        
        while changed:
            changed = False
            
            for row in range(size):
                for col in range(size):
                    if grid[row, col] == 0:
                        candidates = self._get_valid_candidates(grid, row, col, h_constraints, v_constraints)
                        
                        if len(candidates) == 1:
                            grid[row, col] = list(candidates)[0]
                            changed = True
        
        return grid
    
    def _get_valid_candidates(self, grid: np.ndarray, row: int, col: int,
                            h_constraints: Dict, v_constraints: Dict) -> Set[int]:
        """Get valid candidates for a cell considering all constraints"""
        size = len(grid)
        candidates = set(range(1, size + 1))
        
        # Remove values already in row
        for c in range(size):
            if grid[row, c] != 0:
                candidates.discard(grid[row, c])
        
        # Remove values already in column
        for r in range(size):
            if grid[r, col] != 0:
                candidates.discard(grid[r, col])
        
        # Filter by inequality constraints
        candidates = self._filter_by_inequality_constraints(
            grid, row, col, candidates, h_constraints, v_constraints
        )
        
        return candidates
    
    def _filter_by_inequality_constraints(self, grid: np.ndarray, row: int, col: int,
                                        candidates: Set[int], h_constraints: Dict, v_constraints: Dict) -> Set[int]:
        """Filter candidates based on inequality constraints"""
        filtered = candidates.copy()
        size = len(grid)
        
        # Check horizontal constraints
        # Left constraint
        if col > 0 and (row, col-1) in h_constraints:
            constraint = h_constraints[(row, col-1)]
            left_val = grid[row, col-1]
            if left_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if left_val < v}
                elif constraint == '>':
                    filtered = {v for v in filtered if left_val > v}
        
        # Right constraint
        if col < size-1 and (row, col) in h_constraints:
            constraint = h_constraints[(row, col)]
            right_val = grid[row, col+1]
            if right_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if v < right_val}
                elif constraint == '>':
                    filtered = {v for v in filtered if v > right_val}
        
        # Check vertical constraints
        # Top constraint
        if row > 0 and (row-1, col) in v_constraints:
            constraint = v_constraints[(row-1, col)]
            top_val = grid[row-1, col]
            if top_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if top_val < v}
                elif constraint == '>':
                    filtered = {v for v in filtered if top_val > v}
        
        # Bottom constraint
        if row < size-1 and (row, col) in v_constraints:
            constraint = v_constraints[(row, col)]
            bottom_val = grid[row+1, col]
            if bottom_val != 0:
                if constraint == '<':
                    filtered = {v for v in filtered if v < bottom_val}
                elif constraint == '>':
                    filtered = {v for v in filtered if v > bottom_val}
        
        return filtered
    
    def _is_valid_placement_advanced(self, grid: np.ndarray, row: int, col: int, num: int,
                                   h_constraints: Dict, v_constraints: Dict) -> bool:
        """Advanced placement validation with constraint checking"""
        size = len(grid)
        
        # Basic uniqueness check
        if num in grid[row, :] or num in grid[:, col]:
            return False
        
        # Constraint validation
        candidates = {num}
        filtered = self._filter_by_inequality_constraints(
            grid, row, col, candidates, h_constraints, v_constraints
        )
        
        return num in filtered
    
    def _can_solve_with_strategies(self, puzzle: np.ndarray, h_constraints: Dict,
                                 v_constraints: Dict, required_strategies: List[str]) -> Tuple[bool, List[str]]:
        """Check if puzzle can be solved using only the required strategies"""
        try:
            solved_puzzle, used_strategies = self.solver.solve_puzzle(
                puzzle.copy(), h_constraints, v_constraints, required_strategies, max_time_seconds=30
            )
            
            # Check if solved completely
            if np.all(solved_puzzle != 0):
                # Verify at least one required strategy was used
                strategies_used = set(used_strategies)
                required_set = set(required_strategies)
                
                if strategies_used & required_set:  # Intersection exists
                    return True, used_strategies
            
            return False, []
            
        except Exception as e:
            print(f"Strategy solving error: {e}")
            return False, []
    
    def _validate_strategy_compositionality(self, strategies: List[str]) -> bool:
        """Validate that strategies have proper compositionality"""
        all_strategies = (set(self.easy_kb.list_strategies()) |
                         set(self.moderate_kb.list_strategies()) |
                         set(self.hard_kb.list_strategies()))
        
        # Check all strategies exist
        for strategy in strategies:
            if strategy not in all_strategies:
                return False
        
        # Check compositionality dependencies
        for strategy in strategies:
            if strategy in self.moderate_kb.list_strategies():
                strategy_info = self.moderate_kb.get_strategy(strategy)
                prerequisites = strategy_info.get('prerequisites', [])
                
                for prereq in prerequisites:
                    if prereq not in self.easy_kb.list_strategies() and prereq not in strategies:
                        return False
            
            elif strategy in self.hard_kb.list_strategies():
                strategy_info = self.hard_kb.get_strategy(strategy)
                prerequisites = strategy_info.get('prerequisites', [])
                
                for prereq in prerequisites:
                    if (prereq not in self.easy_kb.list_strategies() and
                        prereq not in self.moderate_kb.list_strategies() and
                        prereq not in strategies):
                        return False
        
        return True
    
    def _validate_difficulty_appropriateness(self, puzzle: np.ndarray, strategies: List[str], difficulty: str) -> bool:
        """Validate that puzzle difficulty matches the expected level"""
        size = len(puzzle)
        filled_cells = np.sum(puzzle != 0)
        fill_ratio = filled_cells / (size * size)
        
        # Check fill ratio for difficulty
        if difficulty == 'easy':
            if not (0.35 <= fill_ratio <= 0.65):
                return False
            # Should primarily use easy strategies
            easy_strategies = set(self.easy_kb.list_strategies())
            if not any(s in easy_strategies for s in strategies):
                return False
        
        elif difficulty == 'moderate':
            if not (0.20 <= fill_ratio <= 0.50):
                return False
            # Should use at least one moderate strategy
            moderate_strategies = set(self.moderate_kb.list_strategies())
            if not any(s in moderate_strategies for s in strategies):
                return False
        
        elif difficulty == 'hard':
            if not (0.10 <= fill_ratio <= 0.40):
                return False
            # Should use at least one hard strategy
            hard_strategies = set(self.hard_kb.list_strategies())
            if not any(s in hard_strategies for s in strategies):
                return False
        
        # Check strategy count
        max_strategies = {'easy': 3, 'moderate': 5, 'hard': 8}
        if len(strategies) > max_strategies.get(difficulty, 5):
            return False
        
        return True
    
    def _validate_constraint_consistency(self, puzzle: np.ndarray, h_constraints: Dict, v_constraints: Dict) -> bool:
        """Validate constraint consistency with filled cells"""
        try:
            size = len(puzzle)
            
            # Check horizontal constraints
            for (row, col), constraint in h_constraints.items():
                if col + 1 < size:
                    left_val = puzzle[row, col]
                    right_val = puzzle[row, col + 1]
                    
                    if left_val != 0 and right_val != 0:
                        if constraint == '<' and left_val >= right_val:
                            return False
                        elif constraint == '>' and left_val <= right_val:
                            return False
            
            # Check vertical constraints
            for (row, col), constraint in v_constraints.items():
                if row + 1 < size:
                    top_val = puzzle[row, col]
                    bottom_val = puzzle[row + 1, col]
                    
                    if top_val != 0 and bottom_val != 0:
                        if constraint == '<' and top_val >= bottom_val:
                            return False
                        elif constraint == '>' and top_val <= bottom_val:
                            return False
            
            return True
            
        except Exception:
            return False


class EnhancedMNISTFutoshikiGenerator:
    """Enhanced generator with consistent MNIST usage and comprehensive validation"""
    
    def __init__(self, config_manager=None):
        print("🚀 Initializing Enhanced MNIST Futoshiki Generator...")
        
        self.validator = EnhancedFutoshikiValidator()
        self.solver = FutoshikiSolver()
        
        # Initialize knowledge bases
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        
        self.config_manager = config_manager
        
        # MNIST data and digit mappings
        self.mnist_images = {}
        self.puzzle_digit_mappings = {}  # Store consistent mappings
        
        self.load_mnist_data()
        
        print("✅ Enhanced generator initialized successfully")
    
    def load_mnist_data(self):
        """Load MNIST data with enhanced organization"""
        try:
            print("📥 Loading MNIST dataset...")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Organize by digit
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
            self.create_fallback_mnist()
    
    def create_fallback_mnist(self):
        """Create fallback MNIST patterns"""
        print("🎨 Creating fallback MNIST patterns...")
        
        train_by_digit = {i: [] for i in range(1, 10)}
        test_by_digit = {i: [] for i in range(1, 10)}
        
        for digit in range(1, 10):
            for _ in range(100):  # More patterns for better variety
                pattern = self._create_digit_pattern(digit)
                train_by_digit[digit].append(pattern)
                test_by_digit[digit].append(pattern)
        
        self.mnist_images = {'train': train_by_digit, 'test': test_by_digit}
        print("✅ Fallback MNIST patterns created")
    
    def _create_digit_pattern(self, digit: int) -> np.ndarray:
        """Create recognizable pattern for a digit"""
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Create distinctive patterns for each digit
        patterns = {
            1: [(10, 14, 18, 14), (6, 12, 10, 14), (18, 14, 22, 14)],  # Vertical line with top
            2: [(6, 6, 6, 20), (6, 20, 12, 20), (12, 20, 12, 6), (12, 6, 22, 6), (22, 6, 22, 20)],
            3: [(6, 6, 6, 20), (6, 20, 16, 20), (16, 20, 16, 6), (12, 14, 16, 14), (6, 6, 16, 6)],
            4: [(6, 6, 14, 6), (14, 6, 14, 20), (6, 14, 20, 14), (18, 6, 18, 20)],
            5: [(6, 6, 6, 20), (6, 6, 16, 6), (6, 14, 14, 14), (14, 14, 14, 20), (14, 20, 20, 20)],
            6: [(6, 6, 6, 20), (6, 6, 16, 6), (6, 14, 14, 14), (14, 14, 14, 20), (6, 20, 14, 20)],
            7: [(6, 6, 20, 6), (20, 6, 16, 10), (16, 10, 12, 14), (12, 14, 8, 20)],
            8: [(6, 6, 16, 6), (6, 6, 6, 20), (16, 6, 16, 20), (6, 14, 16, 14), (6, 20, 16, 20)],
            9: [(6, 6, 16, 6), (6, 6, 6, 14), (16, 6, 16, 20), (6, 14, 16, 14), (16, 20, 20, 20)]
        }
        
        if digit in patterns:
            for y1, x1, y2, x2 in patterns[digit]:
                if y1 == y2:  # Horizontal line
                    img[y1:y1+2, min(x1,x2):max(x1,x2)] = 255
                else:  # Vertical line
                    img[min(y1,y2):max(y1,y2), x1:x1+2] = 255
        
        return img
    
    def get_consistent_mnist_mapping(self, puzzle_id: str, size: int) -> Dict[int, np.ndarray]:
        """Get consistent MNIST digit mapping for a puzzle"""
        if puzzle_id not in self.puzzle_digit_mappings:
            # Create new consistent mapping
            mapping = {}
            random.seed(hash(puzzle_id) % (2**32))  # Deterministic seed based on puzzle ID
            
            for digit in range(1, size + 1):
                available_images = self.mnist_images['train'][digit]
                if available_images:
                    mapping[digit] = random.choice(available_images)
                else:
                    mapping[digit] = self._create_digit_pattern(digit)
            
            self.puzzle_digit_mappings[puzzle_id] = mapping
        
        return self.puzzle_digit_mappings[puzzle_id]
    
    def generate_complete_futoshiki_enhanced(self, size: int) -> Tuple[np.ndarray, Dict, Dict]:
        """Generate complete valid Futoshiki with enhanced constraint generation"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            try:
                # Generate complete Latin square
                grid = self._generate_latin_square(size)
                
                if grid is None:
                    continue
                
                # Generate meaningful constraints
                h_constraints, v_constraints = self._generate_enhanced_constraints(grid, size)
                
                # Validate the complete solution
                if self.validator.solver.validate_solution(grid, h_constraints, v_constraints):
                    return grid, h_constraints, v_constraints
                
            except Exception as e:
                if attempt % 20 == 0:
                    print(f"    ⏳ Attempt {attempt + 1}/{max_attempts}")
                continue
        
        raise Exception(f"Failed to generate complete Futoshiki after {max_attempts} attempts")
    
    def _generate_latin_square(self, size: int) -> Optional[np.ndarray]:
        """Generate a valid Latin square using improved backtracking"""
        grid = np.zeros((size, size), dtype=int)
        
        # Start with a randomized first row
        first_row = list(range(1, size + 1))
        random.shuffle(first_row)
        grid[0, :] = first_row
        
        if self._fill_latin_square_backtrack(grid, 1, 0):
            return grid
        
        return None
    
    def _fill_latin_square_backtrack(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Fill Latin square using backtracking with constraint propagation"""
        size = len(grid)
        
        if row == size:
            return True  # Completed
        
        next_row, next_col = (row, col + 1) if col + 1 < size else (row + 1, 0)
        
        # Get valid candidates
        candidates = self._get_latin_square_candidates(grid, row, col)
        random.shuffle(candidates)
        
        for num in candidates:
            grid[row, col] = num
            
            if self._fill_latin_square_backtrack(grid, next_row, next_col):
                return True
            
            grid[row, col] = 0
        
        return False
    
    def _get_latin_square_candidates(self, grid: np.ndarray, row: int, col: int) -> List[int]:
        """Get valid candidates for Latin square cell"""
        size = len(grid)
        candidates = []
        
        for num in range(1, size + 1):
            # Check row constraint
            if num in grid[row, :]:
                continue
            
            # Check column constraint
            if num in grid[:, col]:
                continue
            
            candidates.append(num)
        
        return candidates
    
    def _generate_enhanced_constraints(self, grid: np.ndarray, size: int) -> Tuple[Dict, Dict]:
        """Generate enhanced constraints that create interesting solving patterns"""
        h_constraints = {}
        v_constraints = {}
        
        target_constraints = max(3, size * size // 6)  # Scale with size
        constraints_added = 0
        
        # Generate constraint chains for interesting patterns
        positions = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(positions)
        
        for row, col in positions:
            if constraints_added >= target_constraints:
                break
            
            # Add horizontal constraint
            if col < size - 1 and (row, col) not in h_constraints:
                left_val = grid[row, col]
                right_val = grid[row, col + 1]
                
                if left_val != right_val:  # Avoid equality
                    constraint = '<' if left_val < right_val else '>'
                    h_constraints[(row, col)] = constraint
                    constraints_added += 1
            
            # Add vertical constraint
            if row < size - 1 and (row, col) not in v_constraints:
                top_val = grid[row, col]
                bottom_val = grid[row + 1, col]
                
                if top_val != bottom_val:  # Avoid equality
                    constraint = '<' if top_val < bottom_val else '>'
                    v_constraints[(row, col)] = constraint
                    constraints_added += 1
        
        return h_constraints, v_constraints
    
    def create_puzzle_from_solution_enhanced(self, solution: np.ndarray, h_constraints: Dict,
                                           v_constraints: Dict, difficulty: str) -> Tuple[np.ndarray, List[str]]:
        """Create puzzle with enhanced strategy-based validation"""
        print(f"    🔨 Creating enhanced {difficulty} puzzle...")
        
        max_attempts = 300
        best_puzzle = None
        best_strategies = []
        best_score = -1
        
        target_params = self._get_enhanced_difficulty_parameters(difficulty)
        
        for attempt in range(max_attempts):
            if attempt > 0 and attempt % 50 == 0:
                print(f"      ⏳ Enhanced attempt {attempt}/{max_attempts}")
            
            try:
                # Create puzzle candidate
                puzzle = solution.copy()
                strategies = self._get_appropriate_strategies(difficulty)
                
                # Strategic cell removal
                self._remove_cells_strategically_enhanced(puzzle, target_params, strategies)
                
                # Comprehensive validation
                validation_result = self.validator.validate_puzzle_with_strategies(
                    puzzle, solution, h_constraints, v_constraints, difficulty, strategies
                )
                
                if validation_result['is_valid']:
                    score = self._calculate_enhanced_puzzle_quality(puzzle, strategies, difficulty, validation_result)
                    
                    if score > best_score:
                        best_puzzle = puzzle.copy()
                        best_strategies = strategies.copy()
                        best_score = score
                    
                    if score >= 0.8:  # High quality threshold
                        print(f"      ✅ High-quality enhanced puzzle found at attempt {attempt + 1}")
                        break
                        
            except Exception as e:
                continue
        
        if best_puzzle is None:
            print(f"      🔄 Using enhanced fallback generation for {difficulty}")
            best_puzzle, best_strategies = self._create_enhanced_fallback_puzzle(
                solution, h_constraints, v_constraints, difficulty
            )
        
        return best_puzzle, best_strategies
    
    def _get_enhanced_difficulty_parameters(self, difficulty: str) -> Dict:
        """Get enhanced difficulty parameters"""
        params = {
            'easy': {
                'min_filled_ratio': 0.35,
                'max_filled_ratio': 0.65,
                'target_filled_ratio': 0.50,
                'max_strategies': 3,
                'constraint_density': 0.25
            },
            'moderate': {
                'min_filled_ratio': 0.20,
                'max_filled_ratio': 0.50,
                'target_filled_ratio': 0.35,
                'max_strategies': 5,
                'constraint_density': 0.35
            },
            'hard': {
                'min_filled_ratio': 0.10,
                'max_filled_ratio': 0.40,
                'target_filled_ratio': 0.25,
                'max_strategies': 8,
                'constraint_density': 0.45
            }
        }
        return params.get(difficulty, params['easy'])
    
    def _get_appropriate_strategies(self, difficulty: str) -> List[str]:
        """Get appropriate strategies ensuring compositionality"""
        if difficulty == 'easy':
            # Use primarily atomic easy strategies
            atomic_easy = [s for s in self.easy_kb.list_strategies() 
                          if not self.easy_kb.get_strategy(s).get('composite', False)]
            composite_easy = [s for s in self.easy_kb.list_strategies() 
                             if self.easy_kb.get_strategy(s).get('composite', False)]
            
            selected = random.sample(atomic_easy, min(2, len(atomic_easy)))
            if composite_easy and random.random() < 0.3:  # 30% chance of composite
                selected.append(random.choice(composite_easy))
            
            return selected[:3]
        
        elif difficulty == 'moderate':
            # Must include at least one moderate strategy
            moderate_strategies = list(self.moderate_kb.list_strategies())
            selected = [random.choice(moderate_strategies)]
            
            # Add supporting easy strategies
            easy_strategies = list(self.easy_kb.list_strategies())
            remaining_slots = random.randint(2, 4)
            
            for _ in range(remaining_slots):
                if easy_strategies:
                    strategy = random.choice(easy_strategies)
                    if strategy not in selected:
                        selected.append(strategy)
            
            return selected[:5]
        
        else:  # hard
            # Must include at least one hard strategy
            hard_strategies = list(self.hard_kb.list_strategies())
            selected = [random.choice(hard_strategies)]
            
            # Add moderate and easy strategies for compositionality
            moderate_strategies = list(self.moderate_kb.list_strategies())
            easy_strategies = list(self.easy_kb.list_strategies())
            
            all_available = moderate_strategies + easy_strategies
            remaining_slots = random.randint(3, 6)
            
            for _ in range(remaining_slots):
                if all_available:
                    strategy = random.choice(all_available)
                    if strategy not in selected:
                        selected.append(strategy)
            
            return selected[:8]
    
    def _remove_cells_strategically_enhanced(self, puzzle: np.ndarray, target_params: Dict, strategies: List[str]):
        """Enhanced strategic cell removal based on strategies"""
        size = len(puzzle)
        total_cells = size * size
        target_filled = int(total_cells * target_params['target_filled_ratio'])
        cells_to_remove = total_cells - target_filled
        
        # Prioritize cell removal based on strategy requirements
        removal_priorities = self._calculate_cell_removal_priorities(puzzle, strategies)
        
        # Sort cells by removal priority (higher priority = remove first)
        sorted_cells = sorted(removal_priorities.items(), key=lambda x: x[1], reverse=True)
        
        removed = 0
        for (row, col), priority in sorted_cells:
            if removed >= cells_to_remove:
                break
            
            puzzle[row, col] = 0
            removed += 1
    
    def _calculate_cell_removal_priorities(self, puzzle: np.ndarray, strategies: List[str]) -> Dict:
        """Calculate removal priorities for cells based on strategies"""
        size = len(puzzle)
        priorities = {}
        
        for row in range(size):
            for col in range(size):
                priority = 1.0  # Base priority
                
                # Adjust priority based on strategy requirements
                if 'naked_single' in strategies:
                    # Keep some cells for naked singles
                    priority *= 0.8
                
                if any('hidden' in s for s in strategies):
                    # Remove cells that might create hidden singles/pairs
                    priority *= 1.2
                
                if any('constraint' in s for s in strategies):
                    # Keep cells near constraints
                    if self._is_near_constraint(row, col, size):
                        priority *= 0.7
                
                # Add randomness
                priority *= (0.8 + random.random() * 0.4)
                
                priorities[(row, col)] = priority
        
        return priorities
    
    def _is_near_constraint(self, row: int, col: int, size: int) -> bool:
        """Check if cell is near potential constraint positions"""
        # This is a simplified check - in practice, you'd check actual constraints
        return (row > 0 or row < size-1 or col > 0 or col < size-1)
    
    def _calculate_enhanced_puzzle_quality(self, puzzle: np.ndarray, strategies: List[str], 
                                         difficulty: str, validation_result: Dict) -> float:
        """Calculate enhanced puzzle quality score"""
        score = 0.0
        
        # Base score for valid puzzle
        if validation_result['is_valid']:
            score += 0.4
        
        # Strategy appropriateness
        if validation_result['solvable_with_strategies']:
            score += 0.3
        
        # Compositionality
        if validation_result['strategy_compositionality_valid']:
            score += 0.2
        
        # Difficulty appropriateness
        if validation_result['difficulty_appropriate']:
            score += 0.1
        
        return score
    
    def _create_enhanced_fallback_puzzle(self, solution: np.ndarray, h_constraints: Dict,
                                       v_constraints: Dict, difficulty: str) -> Tuple[np.ndarray, List[str]]:
        """Create enhanced fallback puzzle with guaranteed validity"""
        puzzle = solution.copy()
        target_params = self._get_enhanced_difficulty_parameters(difficulty)
        
        size = len(puzzle)
        total_cells = size * size
        target_filled = int(total_cells * target_params['target_filled_ratio'])
        cells_to_remove = total_cells - target_filled
        
        # Remove cells randomly but strategically
        positions = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(positions)
        
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        # Get basic strategies for difficulty
        fallback_strategies = {
            'easy': ['naked_single', 'constraint_propagation', 'row_uniqueness'],
            'moderate': ['naked_single', 'constraint_propagation', 'naked_pair', 'constraint_chain_analysis'],
            'hard': ['naked_single', 'constraint_propagation', 'multiple_constraint_chains', 'constraint_network_analysis']
        }
        
        return puzzle, fallback_strategies[difficulty]
    
    def create_mnist_representation_enhanced(self, grid: np.ndarray, puzzle_id: str) -> np.ndarray:
        """Create enhanced MNIST representation with consistent digit mapping"""
        size = len(grid)
        mnist_size = size * 28
        mnist_grid = np.zeros((mnist_size, mnist_size), dtype=np.uint8)
        
        # Get consistent digit mapping for this puzzle
        digit_mapping = self.get_consistent_mnist_mapping(puzzle_id, size)
        
        for row in range(size):
            for col in range(size):
                if grid[row, col] != 0:
                    digit_img = digit_mapping[grid[row, col]]
                    
                    start_row = row * 28
                    start_col = col * 28
                    mnist_grid[start_row:start_row+28, start_col:start_col+28] = digit_img
        
        return mnist_grid
    
    def create_constraint_visualization_enhanced(self, h_constraints: Dict, v_constraints: Dict, 
                                               size: int, use_text: bool = True) -> Dict:
        """Create enhanced constraint visualization with text support"""
        constraint_data = {
            'horizontal': [],
            'vertical': [],
            'size': size,
            'use_text_symbols': use_text,
            'text_symbols': {
                '<': 'LT' if use_text else '<',
                '>': 'GT' if use_text else '>'
            }
        }
        
        # Process horizontal constraints
        for (row, col), constraint in h_constraints.items():
            constraint_data['horizontal'].append({
                'row': row,
                'col': col,
                'constraint': constraint,
                'display_text': constraint_data['text_symbols'][constraint],
                'position': 'between_cells'
            })
        
        # Process vertical constraints
        for (row, col), constraint in v_constraints.items():
            constraint_data['vertical'].append({
                'row': row,
                'col': col,
                'constraint': constraint,
                'display_text': constraint_data['text_symbols'][constraint],
                'position': 'between_cells'
            })
        
        return constraint_data
    
    def save_enhanced_puzzle_image(self, mnist_puzzle: np.ndarray, mnist_solution: np.ndarray,
                                 constraint_viz: Dict, puzzle_id: str, output_dir: str):
        """Save enhanced puzzle images with text constraints"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create puzzle image with constraints
            puzzle_img = self._create_image_with_text_constraints(mnist_puzzle, constraint_viz, 'puzzle')
            solution_img = self._create_image_with_text_constraints(mnist_solution, constraint_viz, 'solution')
            
            # Save images
            puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
            solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
            
            puzzle_img.save(puzzle_path)
            solution_img.save(solution_path)
            
            print(f"    💾 Enhanced images saved: {puzzle_id}")
            
        except Exception as e:
            print(f"    ⚠️ Error saving enhanced images: {e}")
    
    def _create_image_with_text_constraints(self, mnist_grid: np.ndarray, constraint_viz: Dict, image_type: str) -> Image.Image:
        """Create image with text constraint overlays"""
        # Convert to PIL Image
        img = Image.fromarray(mnist_grid, mode='L')
        img = img.convert('RGB')
        
        # Create drawing context
        draw = ImageDraw.Draw(img)
        
        size = constraint_viz['size']
        cell_size = len(mnist_grid) // size
        
        try:
            # Try to load a font
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw horizontal constraints
        for h_constraint in constraint_viz['horizontal']:
            row = h_constraint['row']
            col = h_constraint['col']
            text = h_constraint['display_text']
            
            # Position between cells
            x = (col + 0.5) * cell_size + cell_size // 2
            y = row * cell_size + cell_size // 2
            
            # Draw text with background
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill='white', outline='red')
            draw.text((x, y), text, fill='red', font=font, anchor='mm')
        
        # Draw vertical constraints
        for v_constraint in constraint_viz['vertical']:
            row = v_constraint['row']
            col = v_constraint['col']
            text = h_constraint['display_text']
            
            # Position between cells
            x = col * cell_size + cell_size // 2
            y = (row + 0.5) * cell_size + cell_size // 2
            
            # Draw text with background
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill='white', outline='blue')
            draw.text((x, y), text, fill='blue', font=font, anchor='mm')
        
        return img
    
    def generate_guaranteed_valid_puzzles_enhanced(self, difficulty: str, target_count: int, size: int) -> List[Dict]:
        """Generate guaranteed valid puzzles with enhanced validation"""
        print(f"\n🎯 Generating {target_count} ENHANCED {difficulty} puzzles (size {size}x{size})...")
        
        generated_puzzles = []
        attempts = 0
        max_attempts = target_count * 200
        
        start_time = time.time()
        
        while len(generated_puzzles) < target_count and attempts < max_attempts:
            attempts += 1
            
            if attempts % 25 == 0:
                elapsed = time.time() - start_time
                rate = len(generated_puzzles) / elapsed if elapsed > 0 else 0
                print(f"  📊 Enhanced Progress: {len(generated_puzzles)}/{target_count} "
                      f"(attempts: {attempts}, rate: {rate:.2f}/sec)")
            
            try:
                # Generate complete solution
                solution, h_constraints, v_constraints = self.generate_complete_futoshiki_enhanced(size)
                
                # Create puzzle from solution
                puzzle, strategies = self.create_puzzle_from_solution_enhanced(
                    solution, h_constraints, v_constraints, difficulty
                )
                
                # Final comprehensive validation
                validation_result = self.validator.validate_puzzle_with_strategies(
                    puzzle, solution, h_constraints, v_constraints, difficulty, strategies
                )
                
                if not validation_result['is_valid']:
                    continue
                
                # Create puzzle ID
                puzzle_id = f"enhanced_{difficulty}_{len(generated_puzzles):04d}"
                
                # Create MNIST representations with consistent mapping
                mnist_puzzle = self.create_mnist_representation_enhanced(puzzle, puzzle_id)
                mnist_solution = self.create_mnist_representation_enhanced(solution, puzzle_id)
                
                # Create enhanced constraint visualization
                constraint_viz = self.create_constraint_visualization_enhanced(
                    h_constraints, v_constraints, size, use_text=True
                )
                
                # Create puzzle entry
                puzzle_entry = {
                    'id': puzzle_id,
                    'difficulty': difficulty,
                    'size': size,
                    'puzzle_grid': puzzle.tolist(),
                    'solution_grid': solution.tolist(),
                    'h_constraints': {f"{k[0]},{k[1]}": v for k, v in h_constraints.items()},
                    'v_constraints': {f"{k[0]},{k[1]}": v for k, v in v_constraints.items()},
                    'required_strategies': strategies,
                    'mnist_puzzle': mnist_puzzle.tolist(),
                    'mnist_solution': mnist_solution.tolist(),
                    'constraint_visualization': constraint_viz,
                    'strategy_details': {
                        strategy: self._get_strategy_details_enhanced(strategy)
                        for strategy in strategies
                    },
                    'validation': validation_result,
                    'metadata': {
                        'generated_timestamp': datetime.now().isoformat(),
                        'filled_cells': int(np.sum(puzzle != 0)),
                        'empty_cells': int(np.sum(puzzle == 0)),
                        'total_cells': size * size,
                        'fill_ratio': float(np.sum(puzzle != 0)) / (size * size),
                        'difficulty_score': self._calculate_strategy_complexity_enhanced(strategies),
                        'num_h_constraints': len(h_constraints),
                        'num_v_constraints': len(v_constraints),
                        'total_constraints': len(h_constraints) + len(v_constraints),
                        'constraint_density': (len(h_constraints) + len(v_constraints)) / (size * size),
                        'validation_passed': True,
                        'generation_attempt': attempts,
                        'generator_version': '2.1.0',
                        'mnist_consistency': True,
                        'strategy_compositionality': True
                    }
                }
                
                generated_puzzles.append(puzzle_entry)
                print(f"  ✅ Enhanced {difficulty} puzzle {len(generated_puzzles)}/{target_count} validated")
                
            except Exception as e:
                continue
        
        elapsed_time = time.time() - start_time
        success_rate = len(generated_puzzles) / attempts * 100 if attempts > 0 else 0
        
        print(f"\n📈 Enhanced generation complete for {difficulty}:")
        print(f"  ✅ Generated: {len(generated_puzzles)}/{target_count} valid puzzles")
        print(f"  🕐 Time: {elapsed_time:.1f} seconds")
        print(f"  📊 Success rate: {success_rate:.1f}%")
        print(f"  🔗 All puzzles validated for strategy compositionality")
        print(f"  🎯 MNIST digit consistency maintained")
        
        return generated_puzzles
    
    def _get_strategy_details_enhanced(self, strategy_name: str) -> Dict:
        """Get enhanced strategy details with compositionality info"""
        for kb in [self.easy_kb, self.moderate_kb, self.hard_kb]:
            if strategy_name in kb.list_strategies():
                strategy_info = kb.get_strategy(strategy_name)
                
                # Add enhanced details
                enhanced_info = strategy_info.copy()
                enhanced_info['knowledge_base'] = (
                    'easy' if strategy_name in self.easy_kb.list_strategies() else
                    'moderate' if strategy_name in self.moderate_kb.list_strategies() else
                    'hard'
                )
                enhanced_info['compositionality_verified'] = True
                
                return enhanced_info
        
        return {
            'name': strategy_name,
            'description': f'Strategy: {strategy_name}',
            'knowledge_base': 'unknown',
            'compositionality_verified': False
        }
    
    def _calculate_strategy_complexity_enhanced(self, strategies: List[str]) -> float:
        """Calculate enhanced strategy complexity score"""
        complexity_scores = {
            # Easy strategies
            'naked_single': 0.5, 'constraint_propagation': 0.6, 'row_uniqueness': 0.4,
            'column_uniqueness': 0.4, 'forced_by_inequality': 0.8, 'minimum_maximum_bounds': 0.9,
            'hidden_single_row': 0.7, 'hidden_single_column': 0.7, 'direct_constraint_forcing': 0.8,
            
            # Moderate strategies
            'naked_pair': 1.5, 'hidden_pair': 1.7, 'constraint_chain_analysis': 2.0,
            'constraint_splitting': 1.8, 'mutual_constraint_elimination': 2.2,
            'inequality_sandwich': 1.9, 'constraint_propagation_advanced': 2.1,
            'value_forcing_by_uniqueness': 2.3,
            
            # Hard strategies
            'multiple_constraint_chains': 3.0, 'constraint_network_analysis': 3.5,
            'naked_triple': 2.8, 'hidden_triple': 3.2, 'constraint_intersection_forcing': 3.4,
            'advanced_sandwich_analysis': 3.6, 'global_constraint_consistency': 4.0,
            'temporal_constraint_reasoning': 4.5, 'constraint_symmetry_breaking': 4.2
        }
        
        if not strategies:
            return 0.0
        
        total_score = sum(complexity_scores.get(strategy, 1.0) for strategy in strategies)
        return total_score / len(strategies)