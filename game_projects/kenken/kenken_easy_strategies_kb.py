# kenken_easy_strategies_kb.py
"""
Knowledge Base for Easy Ken Ken Solving Strategies
Contains First-Order Logic (FOL) rules for basic Ken Ken solving techniques
"""
from typing import List, Dict

class KenKenEasyStrategiesKB:
    def __init__(self):
        self.strategies = {
            "single_cell_cage": {
                "name": "Single Cell Cage",
                "description": "A cage with only one cell must contain the target value",
                "fol_rule": """
                ∀cage(c) ∀cell(r,col) ∀target(t):
                    [|cells_in_cage(c)| = 1] ∧ [cell(r,col) ∈ cage(c)] ∧ [cage_target(c) = t]
                    → [assign(cell(r,col), t)]
                """,
                "logic": "If a cage contains only one cell, that cell must equal the cage's target value",
                "complexity": "easy",
                "composite": False
            },
            
            "simple_addition_cage": {
                "name": "Simple Addition Cage",
                "description": "Deduce values in small addition cages with limited possibilities",
                "fol_rule": """
                ∀cage(c) ∀operation(add) ∀target(t) ∀grid_size(n):
                    [cage_operation(c) = add] ∧ [cage_target(c) = t] ∧ [|cells_in_cage(c)| = 2]
                    ∧ [∀v ∈ valid_values: 1 ≤ v ≤ n]
                    → [if only_one_combination_possible(c, t, add) then assign_combination(c)]
                """,
                "logic": "If an addition cage has only one possible combination of values, assign them",
                "complexity": "easy",
                "composite": False
            },
            
            "simple_subtraction_cage": {
                "name": "Simple Subtraction Cage",
                "description": "Solve two-cell subtraction cages",
                "fol_rule": """
                ∀cage(c) ∀operation(subtract) ∀target(t) ∀cells{(r1,c1),(r2,c2)}:
                    [cage_operation(c) = subtract] ∧ [cage_target(c) = t] 
                    ∧ [cells_in_cage(c) = {(r1,c1),(r2,c2)}]
                    → [assign_values_such_that |value(r1,c1) - value(r2,c2)| = t]
                """,
                "logic": "In subtraction cages, assign values where the absolute difference equals the target",
                "complexity": "easy",
                "composite": False
            },
            
            "naked_single": {
                "name": "Naked Single",
                "description": "A cell has only one possible value based on cage and grid constraints",
                "fol_rule": """
                ∀cell(r,c) ∀value(v) ∀grid_size(n):
                    [candidates(cell(r,c)) = {v}] ∧ [1 ≤ v ≤ n]
                    → [assign(cell(r,c), v)]
                """,
                "logic": "If a cell has only one candidate value, assign that value",
                "complexity": "easy", 
                "composite": False
            },
            
            "hidden_single_row": {
                "name": "Hidden Single in Row",
                "description": "A value can only go in one cell within a row",
                "fol_rule": """
                ∀row(r) ∀value(v) ∀cell(r,c):
                    [∀c'≠c: v ∉ candidates(cell(r,c'))] ∧ [v ∈ candidates(cell(r,c))]
                    → [assign(cell(r,c), v)]
                """,
                "logic": "If a value can only be placed in one cell within a row, assign it there",
                "complexity": "easy",
                "composite": False
            },
            
            "hidden_single_column": {
                "name": "Hidden Single in Column", 
                "description": "A value can only go in one cell within a column",
                "fol_rule": """
                ∀column(c) ∀value(v) ∀cell(r,c):
                    [∀r'≠r: v ∉ candidates(cell(r',c))] ∧ [v ∈ candidates(cell(r,c))]
                    → [assign(cell(r,c), v)]
                """,
                "logic": "If a value can only be placed in one cell within a column, assign it there",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_by_row": {
                "name": "Eliminate Candidates by Row",
                "description": "Remove candidates that already exist in the same row",
                "fol_rule": """
                ∀row(r) ∀cell(r,c1) ∀cell(r,c2) ∀value(v):
                    [c1 ≠ c2] ∧ [assigned(cell(r,c1), v)] 
                    → [remove_candidate(cell(r,c2), v)]
                """,
                "logic": "If a value is assigned in a row, remove it from candidates of other cells in that row",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_by_column": {
                "name": "Eliminate Candidates by Column",
                "description": "Remove candidates that already exist in the same column",
                "fol_rule": """
                ∀column(c) ∀cell(r1,c) ∀cell(r2,c) ∀value(v):
                    [r1 ≠ r2] ∧ [assigned(cell(r1,c), v)]
                    → [remove_candidate(cell(r2,c), v)]
                """,
                "logic": "If a value is assigned in a column, remove it from candidates of other cells in that column",
                "complexity": "easy",
                "composite": False
            },
            
            "cage_completion": {
                "name": "Cage Completion",
                "description": "Fill remaining cells when cage has only one possibility left",
                "fol_rule": """
                ∀cage(c) ∀remaining_cells(R) ∀assigned_sum(s) ∀target(t):
                    [|R| = 1] ∧ [∀cell ∈ R: empty(cell)] ∧ [cage_operation(c) = add]
                    ∧ [current_sum(c) = s] ∧ [cage_target(c) = t]
                    → [assign(remaining_cell, t - s)]
                """,
                "logic": "When a cage has one empty cell left, calculate its value from the target and current sum",
                "complexity": "easy",
                "composite": False
            },
            
            "basic_multiplication_cage": {
                "name": "Basic Multiplication Cage",
                "description": "Solve simple multiplication cages with obvious factors",
                "fol_rule": """
                ∀cage(c) ∀operation(multiply) ∀target(t) ∀grid_size(n):
                    [cage_operation(c) = multiply] ∧ [cage_target(c) = t] 
                    ∧ [|cells_in_cage(c)| = 2] ∧ [target_has_unique_factorization_in_range(t, n)]
                    → [assign_unique_factor_combination(c, t)]
                """,
                "logic": "If a multiplication cage has a unique factorization within the grid range, assign those factors",
                "complexity": "easy",
                "composite": False
            },
            
            "basic_division_cage": {
                "name": "Basic Division Cage",
                "description": "Solve simple division cages with two cells",
                "fol_rule": """
                ∀cage(c) ∀operation(divide) ∀target(t) ∀cells{(r1,c1),(r2,c2)}:
                    [cage_operation(c) = divide] ∧ [cage_target(c) = t]
                    ∧ [cells_in_cage(c) = {(r1,c1),(r2,c2)}]
                    → [assign_values_such_that max(values)/min(values) = t]
                """,
                "logic": "In division cages, assign values where the larger divided by smaller equals the target",
                "complexity": "easy",
                "composite": False
            }
        }
    
    def get_strategy(self, strategy_name):
        return self.strategies.get(strategy_name)
    
    def list_strategies(self):
        return list(self.strategies.keys())
    
    def get_all_strategies(self):
        return self.strategies
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            return strategy.get('description', f'Strategy: {strategy_name}')
        return f'Strategy: {strategy_name}'

    def get_strategy_patterns(self, strategy_name: str) -> List[str]:
        """Get applicable patterns for this strategy"""
        if 'row' in strategy_name:
            return ['row', 'horizontal']
        elif 'column' in strategy_name:
            return ['column', 'vertical']
        elif 'cage' in strategy_name:
            return ['cage', 'arithmetic']
        elif 'addition' in strategy_name or 'add' in strategy_name:
            return ['addition', 'sum']
        elif 'subtraction' in strategy_name or 'subtract' in strategy_name:
            return ['subtraction', 'difference']
        elif 'multiplication' in strategy_name or 'multiply' in strategy_name:
            return ['multiplication', 'product']
        elif 'division' in strategy_name or 'divide' in strategy_name:
            return ['division', 'quotient']
        else:
            return ['general', 'basic']

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy and strategy.get('composite', False):
            return strategy.get('composed_of', [])
        
        # For non-composite strategies, return basic prerequisites
        basic_prereqs = ['naked_single', 'single_cell_cage']
        if strategy_name not in basic_prereqs:
            return ['naked_single']
        return []
    
    def get_operations_used(self, strategy_name: str) -> List[str]:
        """Get arithmetic operations used by this strategy"""
        if 'addition' in strategy_name or 'add' in strategy_name:
            return ['add']
        elif 'subtraction' in strategy_name or 'subtract' in strategy_name:
            return ['subtract']
        elif 'multiplication' in strategy_name or 'multiply' in strategy_name:
            return ['multiply']
        elif 'division' in strategy_name or 'divide' in strategy_name:
            return ['divide']
        else:
            return []