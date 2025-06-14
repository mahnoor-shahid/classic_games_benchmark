# kenken_easy_strategies_kb.py
"""
Knowledge Base for Easy KenKen Solving Strategies
Contains First-Order Logic (FOL) rules for basic KenKen solving techniques
KenKen rules: Fill grid with numbers 1-N (where N is grid size) such that:
1. No number repeats in any row or column
2. Each cage (outlined region) contains numbers that satisfy the given operation and target
"""

class EasyKenKenStrategiesKB:
    def __init__(self):
        self.strategies = {
            "single_cell_cage": {
                "name": "Single Cell Cage",
                "description": "A cage with only one cell must contain the target value",
                "fol_rule": """
                ∀cage(c) ∀cell(r,col) ∀target(t):
                    [cage_size(c) = 1] ∧ [cell(r,col) ∈ cage(c)] ∧ [cage_target(c) = t]
                    → [assign(cell(r,col), t)]
                """,
                "logic": "If a cage contains only one cell, that cell must equal the cage's target value",
                "complexity": "easy",
                "composite": False
            },
            
            "two_cell_addition_cage": {
                "name": "Two Cell Addition Cage",
                "description": "Solve two-cell addition cages with limited possibilities",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀target(t) ∀grid_size(n):
                    [cage_size(c) = 2] ∧ [cage_operation(c) = addition] ∧ [cage_target(c) = t]
                    ∧ [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    → [candidates(cell(r1,c1)) = {v1 : ∃v2 ∈ [1,n], v1+v2=t ∧ v1≠v2 ∧ valid_in_context(v1,v2)}]
                """,
                "logic": "For two-cell addition cages, find all valid number pairs that sum to target",
                "complexity": "easy",
                "composite": False
            },
            
            "two_cell_subtraction_cage": {
                "name": "Two Cell Subtraction Cage", 
                "description": "Solve two-cell subtraction cages",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀target(t) ∀grid_size(n):
                    [cage_size(c) = 2] ∧ [cage_operation(c) = subtraction] ∧ [cage_target(c) = t]
                    ∧ [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    → [candidates(cell(r1,c1)) = {v1 : ∃v2 ∈ [1,n], |v1-v2|=t ∧ valid_in_context(v1,v2)}]
                """,
                "logic": "For two-cell subtraction cages, find pairs where absolute difference equals target",
                "complexity": "easy", 
                "composite": False
            },
            
            "two_cell_multiplication_cage": {
                "name": "Two Cell Multiplication Cage",
                "description": "Solve two-cell multiplication cages",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀target(t) ∀grid_size(n):
                    [cage_size(c) = 2] ∧ [cage_operation(c) = multiplication] ∧ [cage_target(c) = t]
                    ∧ [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    → [candidates(cell(r1,c1)) = {v1 : ∃v2 ∈ [1,n], v1×v2=t ∧ valid_in_context(v1,v2)}]
                """,
                "logic": "For two-cell multiplication cages, find pairs that multiply to target",
                "complexity": "easy",
                "composite": False
            },
            
            "two_cell_division_cage": {
                "name": "Two Cell Division Cage",
                "description": "Solve two-cell division cages",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀target(t) ∀grid_size(n):
                    [cage_size(c) = 2] ∧ [cage_operation(c) = division] ∧ [cage_target(c) = t]
                    ∧ [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    → [candidates(cell(r1,c1)) = {v1 : ∃v2 ∈ [1,n], (v1/v2=t ∨ v2/v1=t) ∧ valid_in_context(v1,v2)}]
                """,
                "logic": "For two-cell division cages, find pairs where one divided by other equals target",
                "complexity": "easy",
                "composite": False
            },
            
            "naked_single": {
                "name": "Naked Single",
                "description": "A cell has only one possible value after applying cage constraints",
                "fol_rule": """
                ∀cell(r,c) ∀value(v):
                    [candidates(cell(r,c)) = {v}] → [assign(cell(r,c), v)]
                """,
                "logic": "If a cell has only one candidate value, assign that value to the cell",
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
            
            "cage_completion": {
                "name": "Cage Completion",
                "description": "Fill remaining cells when cage is almost complete",
                "fol_rule": """
                ∀cage(c) ∀cell(r,col):
                    [cells_filled(cage(c)) = cage_size(c) - 1] ∧ [cell(r,col) ∈ cage(c)] ∧ [empty(cell(r,col))]
                    → [assign(cell(r,col), required_value_to_complete_cage(c))]
                """,
                "logic": "When all but one cells in a cage are filled, calculate the required value for the last cell",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_by_row": {
                "name": "Eliminate Candidates by Row",
                "description": "Remove candidates that already exist in the same row",
                "fol_rule": """
                ∀row(r) ∀cell(r,c1) ∀cell(r,c2) ∀value(v):
                    [c1 ≠ c2] ∧ [assigned(cell(r,c1), v)] → [remove_candidate(cell(r,c2), v)]
                """,
                "logic": "If a value is assigned to a cell in a row, remove it from candidates of all other cells in that row",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_by_column": {
                "name": "Eliminate Candidates by Column",
                "description": "Remove candidates that already exist in the same column", 
                "fol_rule": """
                ∀column(c) ∀cell(r1,c) ∀cell(r2,c) ∀value(v):
                    [r1 ≠ r2] ∧ [assigned(cell(r1,c), v)] → [remove_candidate(cell(r2,c), v)]
                """,
                "logic": "If a value is assigned to a cell in a column, remove it from candidates of all other cells in that column",
                "complexity": "easy",
                "composite": False
            },
            
            "simple_cage_arithmetic": {
                "name": "Simple Cage Arithmetic",
                "description": "Apply basic arithmetic constraints within cages",
                "fol_rule": """
                ∀cage(c) ∀operation(op) ∀target(t):
                    [cage_operation(c) = op] ∧ [cage_target(c) = t]
                    → [∀assignment_to_cage(c): evaluate(assignment, op) = t]
                """,
                "logic": "All values assigned to a cage must satisfy the cage's arithmetic constraint",
                "complexity": "easy", 
                "composite": False
            },
            
            "forced_candidate_cage": {
                "name": "Forced Candidate in Cage",
                "description": "When only one arrangement satisfies cage constraints",
                "fol_rule": """
                ∀cage(c) ∀cell(r,col) ∀value(v):
                    [∃!assignment: satisfies_cage_constraint(assignment, cage(c))]
                    ∧ [cell(r,col) ∈ cage(c)] ∧ [assignment[cell(r,col)] = v]
                    → [assign(cell(r,col), v)]
                """,
                "logic": "If only one value arrangement satisfies a cage constraint, assign those values",
                "complexity": "easy",
                "composite": False
            },
            
            "cage_boundary_constraint": {
                "name": "Cage Boundary Constraint",
                "description": "Use cage boundaries to limit candidate propagation",
                "fol_rule": """
                ∀cage(c1) ∀cage(c2) ∀cell(r1,col1) ∀cell(r2,col2) ∀value(v):
                    [cell(r1,col1) ∈ cage(c1)] ∧ [cell(r2,col2) ∈ cage(c2)] ∧ [c1 ≠ c2]
                    ∧ [same_row(r1,r2) ∨ same_column(col1,col2)]
                    ∧ [assigned(cell(r1,col1), v)]
                    → [remove_candidate(cell(r2,col2), v)]
                """,
                "logic": "Values cannot repeat in same row/column even across different cages",
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