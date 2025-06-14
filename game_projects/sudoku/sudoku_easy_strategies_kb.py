# easy_strategies_kb.py
"""
Knowledge Base for Easy Sudoku Solving Strategies
Contains First-Order Logic (FOL) rules for basic Sudoku solving techniques
"""

class EasyStrategiesKB:
    def __init__(self):
        self.strategies = {
            "naked_single": {
                "name": "Naked Single",
                "description": "A cell has only one possible value",
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
            
            "hidden_single_box": {
                "name": "Hidden Single in Box",
                "description": "A value can only go in one cell within a 3x3 box",
                "fol_rule": """
                ∀box(b) ∀value(v) ∀cell(r,c):
                    [cell(r,c) ∈ box(b)] ∧ [∀cell(r',c')∈box(b), (r',c')≠(r,c): v ∉ candidates(cell(r',c'))] 
                    ∧ [v ∈ candidates(cell(r,c))] → [assign(cell(r,c), v)]
                """,
                "logic": "If a value can only be placed in one cell within a 3x3 box, assign it there",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_candidates_row": {
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
            
            "eliminate_candidates_column": {
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
            
            "eliminate_candidates_box": {
                "name": "Eliminate Candidates by Box",
                "description": "Remove candidates that already exist in the same 3x3 box",
                "fol_rule": """
                ∀box(b) ∀cell(r1,c1) ∀cell(r2,c2) ∀value(v):
                    [cell(r1,c1) ∈ box(b)] ∧ [cell(r2,c2) ∈ box(b)] ∧ [(r1,c1) ≠ (r2,c2)] 
                    ∧ [assigned(cell(r1,c1), v)] → [remove_candidate(cell(r2,c2), v)]
                """,
                "logic": "If a value is assigned to a cell in a box, remove it from candidates of all other cells in that box",
                "complexity": "easy",
                "composite": False
            },
            
            "full_house_row": {
                "name": "Full House Row",
                "description": "Fill the last empty cell in a row",
                "fol_rule": """
                ∀row(r) ∀cell(r,c):
                    [|{cell(r,c') : empty(cell(r,c'))}| = 1] ∧ [empty(cell(r,c))] 
                    → [assign(cell(r,c), missing_value_in_row(r))]
                """,
                "logic": "If only one cell is empty in a row, assign the missing value to that cell",
                "complexity": "easy",
                "composite": False
            },
            
            "full_house_column": {
                "name": "Full House Column",
                "description": "Fill the last empty cell in a column",
                "fol_rule": """
                ∀column(c) ∀cell(r,c):
                    [|{cell(r',c) : empty(cell(r',c))}| = 1] ∧ [empty(cell(r,c))] 
                    → [assign(cell(r,c), missing_value_in_column(c))]
                """,
                "logic": "If only one cell is empty in a column, assign the missing value to that cell",
                "complexity": "easy",
                "composite": False
            },
            
            "full_house_box": {
                "name": "Full House Box",
                "description": "Fill the last empty cell in a 3x3 box",
                "fol_rule": """
                ∀box(b) ∀cell(r,c):
                    [cell(r,c) ∈ box(b)] ∧ [|{cell(r',c') : cell(r',c') ∈ box(b) ∧ empty(cell(r',c'))}| = 1] 
                    ∧ [empty(cell(r,c))] → [assign(cell(r,c), missing_value_in_box(b))]
                """,
                "logic": "If only one cell is empty in a box, assign the missing value to that cell",
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