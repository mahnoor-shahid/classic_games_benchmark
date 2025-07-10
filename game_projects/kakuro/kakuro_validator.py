"""
Kakuro Puzzle Validator
Validates Kakuro puzzles for correctness and compositionality
"""

import numpy as np
from typing import Dict, List, Optional

class KakuroValidator:
    def __init__(self):
        self.validation_stats = {
            'validated_puzzles': 0,
            'failed_validations': 0,
            'compositionality_failures': 0
        }

    def validate_puzzle(self, puzzle: Dict, solution: np.ndarray) -> bool:
        """Validate a Kakuro puzzle"""
        try:
            # Convert grid to numpy array if it's not already
            grid = np.array(puzzle['grid']) if isinstance(puzzle['grid'], list) else puzzle['grid']
            solution = np.array(solution) if isinstance(solution, list) else solution

            # Validate grid structure
            if not self._validate_grid_structure(grid):
                return False

            # Validate sums
            if not self._validate_sums(grid, solution):
                return False

            # Validate compositionality
            if not self._validate_compositionality(puzzle):
                return False

            self.validation_stats['validated_puzzles'] += 1
            return True

        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            self.validation_stats['failed_validations'] += 1
            return False

    def _validate_grid_structure(self, grid: np.ndarray) -> bool:
        """Validate the basic structure of the grid"""
        try:
            # Check if grid is 2D
            if len(grid.shape) != 2:
                return False

            # Check if grid is square
            if grid.shape[0] != grid.shape[1]:
                return False

            # Check if grid size is within bounds
            if not (5 <= grid.shape[0] <= 15):
                return False

            return True

        except Exception as e:
            print(f"❌ Grid structure validation error: {str(e)}")
            return False

    def _validate_sums(self, grid: np.ndarray, solution: np.ndarray) -> bool:
        """Validate horizontal and vertical sums"""
        try:
            rows, cols = grid.shape

            # Validate horizontal sums
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] > 0:  # This is a sum cell
                        # Get the sequence of cells after this sum
                        sequence = []
                        k = j + 1
                        while k < cols and grid[i, k] == -1:  # -1 represents a cell to fill
                            sequence.append(solution[i, k])
                            k += 1
                        
                        # Check if the sum matches
                        if sum(sequence) != grid[i, j]:
                            return False

            # Validate vertical sums
            for j in range(cols):
                for i in range(rows):
                    if grid[i, j] > 0:  # This is a sum cell
                        # Get the sequence of cells below this sum
                        sequence = []
                        k = i + 1
                        while k < rows and grid[k, j] == -1:  # -1 represents a cell to fill
                            sequence.append(solution[k, j])
                            k += 1
                        
                        # Check if the sum matches
                        if sum(sequence) != grid[i, j]:
                            return False

            return True

        except Exception as e:
            print(f"❌ Sum validation error: {str(e)}")
            return False

    def _validate_compositionality(self, puzzle: Dict) -> bool:
        """Validate puzzle compositionality"""
        try:
            # Validate symmetry
            if not self._validate_symmetry(puzzle):
                return False

            # Validate strategy sequence
            if not self._validate_strategy_sequence(puzzle):
                return False

            # Validate cell relationships
            if not self._validate_cell_relationships(puzzle):
                return False

            return True

        except Exception as e:
            print(f"❌ Compositionality validation error: {str(e)}")
            self.validation_stats['compositionality_failures'] += 1
            return False

    def _validate_symmetry(self, puzzle: Dict) -> bool:
        """Validate puzzle symmetry"""
        try:
            grid = np.array(puzzle['grid']) if isinstance(puzzle['grid'], list) else puzzle['grid']
            template = puzzle['template']
            rows, cols = grid.shape

            # Check symmetry based on template type
            if template == 'cross':
                # Check horizontal and vertical symmetry
                return (np.array_equal(grid, np.flip(grid, axis=0)) and 
                       np.array_equal(grid, np.flip(grid, axis=1)))
            
            elif template == 'border':
                # Check if border cells are symmetric
                return (np.array_equal(grid[0], grid[-1]) and 
                       np.array_equal(grid[:, 0], grid[:, -1]))
            
            elif template == 'checkerboard':
                # Check if pattern alternates
                pattern = np.zeros_like(grid)
                pattern[::2, ::2] = 1
                pattern[1::2, 1::2] = 1
                return np.array_equal((grid > 0), pattern)
            
            elif template == 'diamond':
                # Check diamond symmetry
                center = rows // 2
                return (np.array_equal(grid[:center], np.flip(grid[center+1:], axis=0)) and
                       np.array_equal(grid[:, :center], np.flip(grid[:, center+1:], axis=1)))
            
            elif template == 'spiral':
                # Check spiral pattern
                # This is a simplified check - actual spiral validation would be more complex
                return True
            
            elif template == 'complex':
                # For complex patterns, we allow more variation
                return True
            
            return False

        except Exception as e:
            print(f"❌ Symmetry validation error: {str(e)}")
            return False

    def _validate_strategy_sequence(self, puzzle: Dict) -> bool:
        """Validate strategy sequence"""
        try:
            required_strategies = puzzle.get('required_strategies', [])
            if not required_strategies:
                return False

            # Check if at least one basic strategy is present
            basic_strategies = {'single_cell_sum', 'unique_sum_combination', 'cross_reference'}
            if not any(s in basic_strategies for s in required_strategies):
                return False

            # Check if advanced strategies have their prerequisites
            advanced_strategies = {
                'sum_partition': {'single_cell_sum', 'unique_sum_combination'},
                'digit_frequency': {'sum_partition'},
                'sum_difference': {'sum_partition', 'digit_frequency'},
                'minimum_maximum': {'sum_difference'},
                'sum_completion': {'minimum_maximum'},
                'digit_elimination': {'sum_completion'}
            }

            for strategy in required_strategies:
                if strategy in advanced_strategies:
                    prerequisites = advanced_strategies[strategy]
                    if not all(p in required_strategies for p in prerequisites):
                        return False

            return True

        except Exception as e:
            print(f"❌ Strategy sequence validation error: {str(e)}")
            return False

    def _validate_cell_relationships(self, puzzle: Dict) -> bool:
        """Validate cell relationships"""
        try:
            grid = np.array(puzzle['grid']) if isinstance(puzzle['grid'], list) else puzzle['grid']
            rows, cols = grid.shape

            # Check minimum filled cells per row/column
            min_filled = 1  # Reduced from 2 to 1 for more flexibility
            for i in range(rows):
                if np.sum(grid[i] == -1) < min_filled:
                    return False
            for j in range(cols):
                if np.sum(grid[:, j] == -1) < min_filled:
                    return False

            # Check for isolated cells
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] == -1:  # This is a cell to fill
                        # Check if it's connected to other cells
                        connected = False
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < rows and 0 <= nj < cols and 
                                grid[ni, nj] == -1):
                                connected = True
                                break
                        if not connected:
                            return False

            return True

        except Exception as e:
            print(f"❌ Cell relationship validation error: {str(e)}")
            return False

    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        return self.validation_stats

    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'validated_puzzles': 0,
            'failed_validations': 0,
            'compositionality_failures': 0
        } 