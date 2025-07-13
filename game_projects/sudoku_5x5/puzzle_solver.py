# 5x5 Sudoku Puzzle Solver (digits 0-4)
"""
Sudoku Puzzle Solver and Validator
Implements the strategies from knowledge bases to solve puzzles
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sudoku_easy_strategies_kb import EasyStrategiesKB
from sudoku_moderate_strategies_kb import ModerateStrategiesKB
from sudoku_hard_strategies_kb import HardStrategiesKB

class SudokuSolver:
    def __init__(self):
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.hard_kb = HardStrategiesKB()
        
        # Initialize candidate tracking
        self.candidates = {}
        self.solved_cells = set()
        
    def solve_puzzle(self, puzzle: np.ndarray, allowed_strategies: List[str] = None, max_time_seconds: int = 30) -> Tuple[np.ndarray, List[str]]:
        """
        Solve puzzle using specified strategies with time limit
        Returns: (solved_puzzle, strategies_used)
        """
        import time
        start_time = time.time()
        
        if allowed_strategies is None:
            allowed_strategies = (self.easy_kb.list_strategies() + 
                                self.moderate_kb.list_strategies() + 
                                self.hard_kb.list_strategies())
        
        working_puzzle = puzzle.copy()
        strategies_used = []
        
        # Initialize candidates
        self.initialize_candidates(working_puzzle)
        
        max_iterations = 100
        iteration = 0
        
        while not self.is_complete(working_puzzle) and iteration < max_iterations:
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                raise TimeoutError(f"Solver timed out after {max_time_seconds} seconds")
            
            iteration += 1
            progress_made = False
            
            # Try each allowed strategy
            for strategy_name in allowed_strategies:
                if self.apply_strategy(working_puzzle, strategy_name):
                    strategies_used.append(strategy_name)
                    progress_made = True
                    break  # Try again from the beginning with updated state
            
            if not progress_made:
                # No more progress possible with allowed strategies
                break
        
        return working_puzzle, strategies_used
    
    def initialize_candidates(self, puzzle: np.ndarray):
        """Initialize candidate sets for each empty cell"""
        self.candidates = {}
        self.solved_cells = set()
        
        for row in range(5):
            for col in range(5):
                if puzzle[row, col] == -1:
                    self.candidates[(row, col)] = self.get_possible_values(puzzle, row, col)
                else:
                    self.solved_cells.add((row, col))
    
    def get_possible_values(self, puzzle: np.ndarray, row: int, col: int) -> Set[int]:
        """Get possible values for a cell"""
        if puzzle[row, col] != -1:
            return set()
        
        possible = set(range(5))
        
        # Remove values in same row
        possible -= set(puzzle[row, :])
        
        # Remove values in same column
        possible -= set(puzzle[:, col])
        
        # Remove values in same 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_values = puzzle[box_row:box_row+3, box_col:box_col+3].flatten()
        possible -= set(box_values)
        
        # Remove -1 (empty cell marker)
        possible.discard(-1)
        
        return possible
    
    def update_candidates_after_assignment(self, puzzle: np.ndarray, row: int, col: int, value: int):
        """Update candidate sets after assigning a value to a cell"""
        # Remove from candidates of this cell
        if (row, col) in self.candidates:
            del self.candidates[(row, col)]
        
        self.solved_cells.add((row, col))
        
        # Update candidates in same row
        for c in range(5):
            if (row, c) in self.candidates:
                self.candidates[(row, c)].discard(value)
        
        # Update candidates in same column
        for r in range(5):
            if (r, col) in self.candidates:
                self.candidates[(r, col)].discard(value)
        
        # Update candidates in same box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) in self.candidates:
                    self.candidates[(r, c)].discard(value)
    
    def is_complete(self, puzzle: np.ndarray) -> bool:
        """Check if puzzle is completely solved"""
        return np.all(puzzle != -1)
    
    def apply_strategy(self, puzzle: np.ndarray, strategy_name: str) -> bool:
        """Apply a specific strategy using knowledge base strategies"""
        
        # Easy strategies
        if strategy_name == "naked_single":
            return self.apply_naked_single(puzzle)
        elif strategy_name == "hidden_single_row":
            return self.apply_hidden_single_row(puzzle)
        elif strategy_name == "hidden_single_column":
            return self.apply_hidden_single_column(puzzle)
        elif strategy_name == "hidden_single_box":
            return self.apply_hidden_single_box(puzzle)
        elif strategy_name == "full_house_row":
            return self.apply_full_house_row(puzzle)
        elif strategy_name == "full_house_column":
            return self.apply_full_house_column(puzzle)
        elif strategy_name == "full_house_box":
            return self.apply_full_house_box(puzzle)
        elif strategy_name in ["eliminate_candidates_row", "eliminate_candidates_column", "eliminate_candidates_box"]:
            return False  # These are handled automatically in candidate updates
        
        # Moderate strategies
        elif strategy_name == "naked_pair":
            return self.apply_naked_pair(puzzle)
        elif strategy_name == "naked_triple":
            return self.apply_naked_triple(puzzle)
        elif strategy_name == "hidden_pair":
            return self.apply_hidden_pair(puzzle)
        elif strategy_name == "hidden_triple":
            return self.apply_hidden_triple(puzzle)
        elif strategy_name == "pointing_pairs":
            return self.apply_pointing_pairs(puzzle)
        elif strategy_name == "box_line_reduction":
            return self.apply_box_line_reduction(puzzle)
        elif strategy_name == "x_wing":
            return self.apply_x_wing(puzzle)
        elif strategy_name == "xy_wing":
            return self.apply_xy_wing(puzzle)
        elif strategy_name == "simple_coloring":
            return self.apply_simple_coloring(puzzle)
        
        # Hard strategies (implementable ones from knowledge base)
        elif strategy_name == "swordfish":
            return self.apply_swordfish(puzzle)
        elif strategy_name == "jellyfish":
            return self.apply_jellyfish(puzzle)
        elif strategy_name == "xyz_wing":
            return self.apply_xyz_wing(puzzle)
        elif strategy_name == "wxyz_wing":
            return self.apply_wxyz_wing(puzzle)
        elif strategy_name == "multi_coloring":
            return self.apply_multi_coloring(puzzle)
        elif strategy_name == "sk_loop":
            return self.apply_sk_loop(puzzle)
        elif strategy_name == "aic_discontinuous":
            return self.apply_aic_discontinuous(puzzle)
        
        # If strategy not implemented, return False (no progress)
        # This allows the system to continue with other strategies
        return False
    
    def apply_hidden_pair(self, puzzle: np.ndarray) -> bool:
        """Apply hidden pair strategy - simplified implementation"""
        progress = False
        
        # Check rows, columns, and boxes for hidden pairs
        for row in range(5):
            progress |= self.find_hidden_pairs_in_unit(
                [(row, col) for col in range(5) if (row, col) in self.candidates], 'row'
            )
        
        for col in range(5):
            progress |= self.find_hidden_pairs_in_unit(
                [(row, col) for row in range(5) if (row, col) in self.candidates], 'col'
            )
        
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                box_cells = [(r, c) for r in range(box_row, box_row + 3) 
                            for c in range(box_col, box_col + 3) if (r, c) in self.candidates]
                progress |= self.find_hidden_pairs_in_unit(box_cells, 'box')
        
        return progress
    
    def find_hidden_pairs_in_unit(self, unit_cells: List[Tuple[int, int]], unit_type: str) -> bool:
        """Find hidden pairs within a unit - simplified implementation"""
        progress = False
        
        # For each pair of values, check if they appear in exactly 2 cells
        for val1 in range(5):
            for val2 in range(val1 + 1, 5):
                cells_with_val1 = [cell for cell in unit_cells if val1 in self.candidates.get(cell, set())]
                cells_with_val2 = [cell for cell in unit_cells if val2 in self.candidates.get(cell, set())]
                
                # Check if both values appear in exactly the same 2 cells
                if (len(cells_with_val1) == 2 and len(cells_with_val2) == 2 and 
                    set(cells_with_val1) == set(cells_with_val2)):
                    
                    # Found hidden pair - remove other candidates from these cells
                    for cell in cells_with_val1:
                        if cell in self.candidates:
                            before_size = len(self.candidates[cell])
                            self.candidates[cell] = {val1, val2}
                            if len(self.candidates[cell]) < before_size:
                                progress = True
        
        return progress
    
    def apply_hidden_triple(self, puzzle: np.ndarray) -> bool:
        """Apply hidden triple strategy - simplified implementation"""
        # Simplified version - similar to hidden pair but for 3 values
        return self.apply_hidden_pair(puzzle)  # Fallback for now
    
    def apply_xy_wing(self, puzzle: np.ndarray) -> bool:
        """Apply XY-Wing strategy - simplified implementation"""
        progress = False
        
        # Look for XY-Wing patterns: pivot cell with 2 candidates, two wing cells
        for pivot_cell, pivot_candidates in self.candidates.items():
            if len(pivot_candidates) == 2:
                pivot_row, pivot_col = pivot_cell
                x, y = list(pivot_candidates)
                
                # Find wing cells that see the pivot
                wing1_cells = []
                wing2_cells = []
                
                # Check row, column, and box for potential wings
                for cell, candidates in self.candidates.items():
                    if len(candidates) == 2 and cell != pivot_cell:
                        if self.cells_see_each_other(pivot_cell, cell):
                            if x in candidates and y not in candidates:
                                wing1_cells.append(cell)
                            elif y in candidates and x not in candidates:
                                wing2_cells.append(cell)
                
                # Try to form XY-Wing
                for wing1 in wing1_cells:
                    for wing2 in wing2_cells:
                        wing1_candidates = self.candidates[wing1]
                        wing2_candidates = self.candidates[wing2]
                        
                        # Find common candidate between wings (not x or y)
                        common = wing1_candidates & wing2_candidates
                        if len(common) == 1:
                            z = list(common)[0]
                            if z != x and z != y:
                                # Found XY-Wing pattern - eliminate z from cells that see both wings
                                progress |= self.eliminate_xy_wing(wing1, wing2, z)
        
        return progress
    
    def cells_see_each_other(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> bool:
        """Check if two cells can see each other (same row, column, or box)"""
        r1, c1 = cell1
        r2, c2 = cell2
        
        # Same row or column
        if r1 == r2 or c1 == c2:
            return True
        
        # Same box
        if (r1 // 3) == (r2 // 3) and (c1 // 3) == (c2 // 3):
            return True
        
        return False
    
    def eliminate_xy_wing(self, wing1: Tuple[int, int], wing2: Tuple[int, int], value: int) -> bool:
        """Eliminate value from cells that see both wings"""
        progress = False
        
        for cell in self.candidates:
            if (cell != wing1 and cell != wing2 and 
                self.cells_see_each_other(cell, wing1) and 
                self.cells_see_each_other(cell, wing2)):
                
                if value in self.candidates[cell]:
                    self.candidates[cell].discard(value)
                    progress = True
        
        return progress
    
    # Hard strategy implementations (simplified versions)
    def apply_jellyfish(self, puzzle: np.ndarray) -> bool:
        """Apply jellyfish strategy - extension of swordfish"""
        return self.apply_swordfish(puzzle)  # Simplified fallback
    
    def apply_xyz_wing(self, puzzle: np.ndarray) -> bool:
        """Apply XYZ-Wing strategy - extension of XY-Wing"""
        return self.apply_xy_wing(puzzle)  # Simplified fallback
    
    def apply_wxyz_wing(self, puzzle: np.ndarray) -> bool:
        """Apply WXYZ-Wing strategy - extension of XYZ-Wing"""
        return self.apply_xy_wing(puzzle)  # Simplified fallback
    
    def apply_multi_coloring(self, puzzle: np.ndarray) -> bool:
        """Apply multi-coloring strategy - extension of simple coloring"""
        return self.apply_simple_coloring(puzzle)  # Simplified fallback
    
    def apply_sk_loop(self, puzzle: np.ndarray) -> bool:
        """Apply SK Loop strategy - simplified implementation"""
        # Very complex strategy - simplified version
        return False  # Skip for now, but strategy is recognized
    
    def apply_aic_discontinuous(self, puzzle: np.ndarray) -> bool:
        """Apply Alternating Inference Chain strategy - simplified implementation"""
        # Very complex strategy - simplified version
        return False  # Skip for now, but strategy is recognized
    
    def apply_naked_triple(self, puzzle: np.ndarray) -> bool:
        """Apply naked triple strategy - simplified implementation"""
        progress = False
        
        # Check rows
        for row in range(5):
            progress |= self.find_naked_triples_in_unit(
                [(row, col) for col in range(5) if (row, col) in self.candidates]
            )
        
        # Check columns
        for col in range(5):
            progress |= self.find_naked_triples_in_unit(
                [(row, col) for row in range(5) if (row, col) in self.candidates]
            )
        
        # Check boxes
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                box_cells = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        if (r, c) in self.candidates:
                            box_cells.append((r, c))
                progress |= self.find_naked_triples_in_unit(box_cells)
        
        return progress
    
    def find_naked_triples_in_unit(self, unit_cells: List[Tuple[int, int]]) -> bool:
        """Find naked triples within a unit - simplified version"""
        progress = False
        
        # Find cells with 2 or 3 candidates
        candidate_cells = [cell for cell in unit_cells 
                          if 2 <= len(self.candidates.get(cell, set())) <= 3]
        
        # Look for triples (simplified approach)
        for i in range(len(candidate_cells)):
            for j in range(i + 1, len(candidate_cells)):
                for k in range(j + 1, len(candidate_cells)):
                    cell1, cell2, cell3 = candidate_cells[i], candidate_cells[j], candidate_cells[k]
                    
                    combined_candidates = (self.candidates[cell1] | 
                                         self.candidates[cell2] | 
                                         self.candidates[cell3])
                    
                    if len(combined_candidates) == 3:
                        # Found a naked triple - eliminate from other cells
                        for cell in unit_cells:
                            if cell not in [cell1, cell2, cell3] and cell in self.candidates:
                                before_size = len(self.candidates[cell])
                                self.candidates[cell] -= combined_candidates
                                if len(self.candidates[cell]) < before_size:
                                    progress = True
        
        return progress
    
    def apply_swordfish(self, puzzle: np.ndarray) -> bool:
        """Apply swordfish strategy - simplified implementation"""
        # Simplified swordfish: just an extension of X-Wing logic
        # In practice, this would be more complex
        return self.apply_x_wing(puzzle)  # Fallback to X-Wing for now
    
    def apply_simple_coloring(self, puzzle: np.ndarray) -> bool:
        """Apply simple coloring strategy - simplified implementation"""
        progress = False
        
        # Very simplified coloring approach
        # Look for cells with only 2 candidates and try to find contradictions
        for (row, col), candidates in list(self.candidates.items()):
            if len(candidates) == 2:
                # Try to find chain patterns (simplified)
                for value in candidates:
                    # Look for other cells in same units with this value
                    related_cells = self.find_related_cells_with_value(row, col, value)
                    if len(related_cells) > 1:
                        # Simple elimination based on coloring logic
                        progress |= self.eliminate_by_coloring(related_cells, value)
        
        return progress
    
    def find_related_cells_with_value(self, row: int, col: int, value: int) -> List[Tuple[int, int]]:
        """Find cells related by coloring chains - simplified"""
        related = []
        
        # Check same row
        for c in range(5):
            if (row, c) in self.candidates and value in self.candidates[(row, c)]:
                related.append((row, c))
        
        # Check same column  
        for r in range(5):
            if (r, col) in self.candidates and value in self.candidates[(r, col)]:
                related.append((r, col))
        
        return related
    
    def eliminate_by_coloring(self, cells: List[Tuple[int, int]], value: int) -> bool:
        """Eliminate candidates based on coloring logic - simplified"""
        # Very basic elimination
        if len(cells) > 2:
            # Remove value from middle cells (simplified approach)
            middle_cell = cells[len(cells)//2]
            if middle_cell in self.candidates and value in self.candidates[middle_cell]:
                self.candidates[middle_cell].discard(value)
                return True
        return False
    
    def apply_box_line_reduction(self, puzzle: np.ndarray) -> bool:
        """Apply box/line reduction strategy - simplified"""
        # This is the reverse of pointing pairs
        return self.apply_pointing_pairs(puzzle)  # Simplified implementation
    
    def apply_naked_single(self, puzzle: np.ndarray) -> bool:
        """Apply naked single strategy"""
        progress = False
        for (row, col), candidates in list(self.candidates.items()):
            if len(candidates) == 1:
                value = list(candidates)[0]
                puzzle[row, col] = value
                self.update_candidates_after_assignment(puzzle, row, col, value)
                progress = True
        return progress
    
    def apply_hidden_single_row(self, puzzle: np.ndarray) -> bool:
        """Apply hidden single in row strategy"""
        progress = False
        for row in range(5):
            # Get all empty cells in this row
            empty_cells = [(row, col) for col in range(5) if (row, col) in self.candidates]
            
            # For each value 1-9, check if it can only go in one cell
            for value in range(5):
                possible_cells = [cell for cell in empty_cells 
                                if value in self.candidates.get(cell, set())]
                
                if len(possible_cells) == 1:
                    cell_row, cell_col = possible_cells[0]
                    puzzle[cell_row, cell_col] = value
                    self.update_candidates_after_assignment(puzzle, cell_row, cell_col, value)
                    progress = True
        
        return progress
    
    def apply_hidden_single_column(self, puzzle: np.ndarray) -> bool:
        """Apply hidden single in column strategy"""
        progress = False
        for col in range(5):
            # Get all empty cells in this column
            empty_cells = [(row, col) for row in range(5) if (row, col) in self.candidates]
            
            # For each value 1-9, check if it can only go in one cell
            for value in range(5):
                possible_cells = [cell for cell in empty_cells 
                                if value in self.candidates.get(cell, set())]
                
                if len(possible_cells) == 1:
                    cell_row, cell_col = possible_cells[0]
                    puzzle[cell_row, cell_col] = value
                    self.update_candidates_after_assignment(puzzle, cell_row, cell_col, value)
                    progress = True
        
        return progress
    
    def apply_hidden_single_box(self, puzzle: np.ndarray) -> bool:
        """Apply hidden single in box strategy"""
        progress = False
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                # Get all empty cells in this box
                empty_cells = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        if (r, c) in self.candidates:
                            empty_cells.append((r, c))
                
                # For each value 1-9, check if it can only go in one cell
                for value in range(5):
                    possible_cells = [cell for cell in empty_cells 
                                    if value in self.candidates.get(cell, set())]
                    
                    if len(possible_cells) == 1:
                        cell_row, cell_col = possible_cells[0]
                        puzzle[cell_row, cell_col] = value
                        self.update_candidates_after_assignment(puzzle, cell_row, cell_col, value)
                        progress = True
        
        return progress
    
    def apply_eliminate_candidates_row(self, puzzle: np.ndarray) -> bool:
        """Apply eliminate candidates by row strategy"""
        # This is typically handled automatically in update_candidates_after_assignment
        return False
    
    def apply_eliminate_candidates_column(self, puzzle: np.ndarray) -> bool:
        """Apply eliminate candidates by column strategy"""
        # This is typically handled automatically in update_candidates_after_assignment
        return False
    
    def apply_eliminate_candidates_box(self, puzzle: np.ndarray) -> bool:
        """Apply eliminate candidates by box strategy"""
        # This is typically handled automatically in update_candidates_after_assignment
        return False
    
    def apply_full_house_row(self, puzzle: np.ndarray) -> bool:
        """Apply full house row strategy"""
        progress = False
        for row in range(5):
            empty_cells = [col for col in range(5) if puzzle[row, col] == -1]
            if len(empty_cells) == 1:
                col = empty_cells[0]
                # Find missing value
                present_values = set(puzzle[row, :]) - {-1}
                missing_value = (set(range(5)) - present_values).pop()
                puzzle[row, col] = missing_value
                self.update_candidates_after_assignment(puzzle, row, col, missing_value)
                progress = True
        return progress
    
    def apply_full_house_column(self, puzzle: np.ndarray) -> bool:
        """Apply full house column strategy"""
        progress = False
        for col in range(5):
            empty_cells = [row for row in range(5) if puzzle[row, col] == -1]
            if len(empty_cells) == 1:
                row = empty_cells[0]
                # Find missing value
                present_values = set(puzzle[:, col]) - {-1}
                missing_value = (set(range(5)) - present_values).pop()
                puzzle[row, col] = missing_value
                self.update_candidates_after_assignment(puzzle, row, col, missing_value)
                progress = True
        return progress
    
    def apply_full_house_box(self, puzzle: np.ndarray) -> bool:
        """Apply full house box strategy"""
        progress = False
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                box = puzzle[box_row:box_row+3, box_col:box_col+3]
                empty_positions = [(r, c) for r in range(3) for c in range(3) if box[r, c] == -1]
                
                if len(empty_positions) == 1:
                    r, c = empty_positions[0]
                    actual_row, actual_col = box_row + r, box_col + c
                    # Find missing value
                    present_values = set(box.flatten()) - {-1}
                    missing_value = (set(range(5)) - present_values).pop()
                    puzzle[actual_row, actual_col] = missing_value
                    self.update_candidates_after_assignment(puzzle, actual_row, actual_col, missing_value)
                    progress = True
        return progress
    
    def apply_naked_pair(self, puzzle: np.ndarray) -> bool:
        """Apply naked pair strategy"""
        progress = False
        
        # Check rows
        for row in range(5):
            progress |= self.find_naked_pairs_in_unit(
                [(row, col) for col in range(5) if (row, col) in self.candidates]
            )
        
        # Check columns
        for col in range(5):
            progress |= self.find_naked_pairs_in_unit(
                [(row, col) for row in range(5) if (row, col) in self.candidates]
            )
        
        # Check boxes
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                box_cells = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        if (r, c) in self.candidates:
                            box_cells.append((r, c))
                progress |= self.find_naked_pairs_in_unit(box_cells)
        
        return progress
    
    def find_naked_pairs_in_unit(self, unit_cells: List[Tuple[int, int]]) -> bool:
        """Find naked pairs within a unit (row, column, or box)"""
        progress = False
        
        # Find cells with exactly 2 candidates
        pair_cells = [cell for cell in unit_cells 
                     if len(self.candidates.get(cell, set())) == 2]
        
        # Check for matching pairs
        for i in range(len(pair_cells)):
            for j in range(i + 1, len(pair_cells)):
                cell1, cell2 = pair_cells[i], pair_cells[j]
                candidates1 = self.candidates[cell1]
                candidates2 = self.candidates[cell2]
                
                if candidates1 == candidates2:
                    # Found a naked pair - eliminate these values from other cells
                    pair_values = candidates1
                    for cell in unit_cells:
                        if cell != cell1 and cell != cell2 and cell in self.candidates:
                            before_size = len(self.candidates[cell])
                            self.candidates[cell] -= pair_values
                            if len(self.candidates[cell]) < before_size:
                                progress = True
        
        return progress
    
    def apply_naked_triple(self, puzzle: np.ndarray) -> bool:
        """Apply naked triple strategy"""
        # Implementation similar to naked_pair but for three cells
        # This is a simplified version
        return False
    
    def apply_hidden_pair(self, puzzle: np.ndarray) -> bool:
        """Apply hidden pair strategy"""
        # Implementation for hidden pairs
        # This is a simplified version
        return False
    
    def apply_pointing_pairs(self, puzzle: np.ndarray) -> bool:
        """Apply pointing pairs strategy"""
        progress = False
        
        # For each box, check if candidates for a value are confined to one row/column
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                for value in range(5):
                    # Find all cells in this box that can contain this value
                    box_candidates = []
                    for r in range(box_row, box_row + 3):
                        for c in range(box_col, box_col + 3):
                            if (r, c) in self.candidates and value in self.candidates[(r, c)]:
                                box_candidates.append((r, c))
                    
                    if len(box_candidates) >= 2:
                        # Check if all candidates are in the same row
                        rows = set(r for r, c in box_candidates)
                        if len(rows) == 1:
                            # All in same row - eliminate from rest of row
                            target_row = list(rows)[0]
                            for col in range(5):
                                if (target_row, col) in self.candidates and col not in range(box_col, box_col + 3):
                                    if value in self.candidates[(target_row, col)]:
                                        self.candidates[(target_row, col)].discard(value)
                                        progress = True
                        
                        # Check if all candidates are in the same column
                        cols = set(c for r, c in box_candidates)
                        if len(cols) == 1:
                            # All in same column - eliminate from rest of column
                            target_col = list(cols)[0]
                            for row in range(5):
                                if (row, target_col) in self.candidates and row not in range(box_row, box_row + 3):
                                    if value in self.candidates[(row, target_col)]:
                                        self.candidates[(row, target_col)].discard(value)
                                        progress = True
        
        return progress
    
    def apply_x_wing(self, puzzle: np.ndarray) -> bool:
        """Apply X-Wing strategy"""
        progress = False
        
        # Check rows for X-Wing patterns
        for value in range(5):
            # Find rows where the value has exactly 2 possible positions
            row_candidates = {}
            for row in range(5):
                positions = [col for col in range(5) 
                           if (row, col) in self.candidates and value in self.candidates[(row, col)]]
                if len(positions) == 2:
                    row_candidates[row] = positions
            
            # Look for X-Wing pattern
            rows = list(row_candidates.keys())
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    row1, row2 = rows[i], rows[j]
                    if row_candidates[row1] == row_candidates[row2]:
                        # Found X-Wing - eliminate from columns
                        col1, col2 = row_candidates[row1]
                        for row in range(5):
                            if row != row1 and row != row2:
                                for col in [col1, col2]:
                                    if (row, col) in self.candidates and value in self.candidates[(row, col)]:
                                        self.candidates[(row, col)].discard(value)
                                        progress = True
        
        return progress
    
    def validate_solution(self, puzzle: np.ndarray) -> bool:
        """Validate if the puzzle solution is correct"""
        # Check if all cells are filled
        if np.any(puzzle == -1):
            return False
        
        # Check rows
        for row in range(5):
            if set(puzzle[row, :]) != set(range(5)):
                return False
        
        # Check columns
        for col in range(5):
            if set(puzzle[:, col]) != set(range(5)):
                return False
        
        # Check boxes
        for box_row in range(0, 5, 3):
            for box_col in range(0, 5, 3):
                box = puzzle[box_row:box_row+3, box_col:box_col+3]
                if set(box.flatten()) != set(range(5)):
                    return False
        
        return True
    
    def get_strategy_application_order(self, difficulty: str) -> List[str]:
        """Get the recommended order for applying strategies based on difficulty"""
        if difficulty == 'easy':
            return [
                'naked_single',
                'hidden_single_row',
                'hidden_single_column', 
                'hidden_single_box',
                'full_house_row',
                'full_house_column',
                'full_house_box'
            ]
        elif difficulty == 'moderate':
            return [
                'naked_single',
                'hidden_single_row',
                'hidden_single_column',
                'hidden_single_box',
                'naked_pair',
                'hidden_pair',
                'pointing_pairs',
                'full_house_row',
                'full_house_column',
                'full_house_box'
            ]
        else:  # hard
            return [
                'naked_single',
                'hidden_single_row',
                'hidden_single_column',
                'hidden_single_box',
                'naked_pair',
                'naked_triple',
                'hidden_pair',
                'pointing_pairs',
                'x_wing',
                'full_house_row',
                'full_house_column',
                'full_house_box'
            ]