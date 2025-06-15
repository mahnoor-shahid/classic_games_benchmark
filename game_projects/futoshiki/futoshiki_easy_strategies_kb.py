# futoshiki_easy_strategies_kb.py
"""
Knowledge Base for Easy Futoshiki Solving Strategies
Contains First-Order Logic (FOL) rules for basic Futoshiki solving techniques
Futoshiki: Latin square with inequality constraints between adjacent cells
"""
from typing import List, Dict

class FutoshikiEasyStrategiesKB:
    def __init__(self):
        self.strategies = {
            "naked_single": {
                "name": "Naked Single",
                "description": "A cell has only one possible value considering constraints",
                "fol_rule": """
                ∀cell(r,c) ∀value(v):
                    [candidates(cell(r,c)) = {v}] → [assign(cell(r,c), v)]
                """,
                "logic": "If a cell has only one candidate value after applying all constraints, assign that value",
                "complexity": "easy",
                "composite": False,
                "prerequisites": [],
                "applies_to": ["empty_cells"]
            },
            
            "constraint_propagation": {
                "name": "Basic Constraint Propagation",
                "description": "Remove candidates that violate inequality constraints",
                "fol_rule": """
                ∀cell(r1,c1) ∀cell(r2,c2) ∀constraint(op):
                    [adjacent(cell(r1,c1), cell(r2,c2))] ∧ [inequality(cell(r1,c1), op, cell(r2,c2))]
                    → [∀v1 ∈ candidates(cell(r1,c1)), v2 ∈ candidates(cell(r2,c2)):
                       ¬satisfies(v1, op, v2) → remove_candidate_pair(cell(r1,c1), v1, cell(r2,c2), v2)]
                """,
                "logic": "Remove candidate values that would violate inequality constraints between adjacent cells",
                "complexity": "easy",
                "composite": False,
                "prerequisites": [],
                "applies_to": ["constraint_pairs"]
            },
            
            "row_uniqueness": {
                "name": "Row Uniqueness Constraint",
                "description": "Each number appears exactly once in each row",
                "fol_rule": """
                ∀row(r) ∀value(v) ∀cell(r,c):
                    [assigned(cell(r,c'), v) ∧ c' ≠ c] → [remove_candidate(cell(r,c), v)]
                """,
                "logic": "If a value is assigned in a row, remove it from candidates of all other cells in that row",
                "complexity": "easy",
                "composite": False,
                "prerequisites": [],
                "applies_to": ["rows"]
            },
            
            "column_uniqueness": {
                "name": "Column Uniqueness Constraint", 
                "description": "Each number appears exactly once in each column",
                "fol_rule": """
                ∀column(c) ∀value(v) ∀cell(r,c):
                    [assigned(cell(r',c), v) ∧ r' ≠ r] → [remove_candidate(cell(r,c), v)]
                """,
                "logic": "If a value is assigned in a column, remove it from candidates of all other cells in that column",
                "complexity": "easy",
                "composite": False,
                "prerequisites": [],
                "applies_to": ["columns"]
            },
            
            "forced_by_inequality": {
                "name": "Forced Value by Inequality",
                "description": "When only one value can satisfy an inequality constraint",
                "fol_rule": """
                ∀cell(r1,c1) ∀cell(r2,c2) ∀constraint(op) ∀value(v):
                    [inequality(cell(r1,c1), op, cell(r2,c2))] 
                    ∧ [|{v' ∈ candidates(cell(r1,c1)) : ∃v'' ∈ candidates(cell(r2,c2)), satisfies(v', op, v'')}| = 1]
                    → [assign(cell(r1,c1), v)]
                """,
                "logic": "If only one candidate in a cell can satisfy an inequality constraint, assign that value",
                "complexity": "easy",
                "composite": True,
                "composed_of": ["constraint_propagation"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["constraint_pairs"]
            },
            
            "minimum_maximum_bounds": {
                "name": "Minimum/Maximum Value Bounds",
                "description": "Determine bounds based on chain of inequalities",
                "fol_rule": """
                ∀cell(r,c) ∀value_set(V):
                    [chain_inequality_min(cell(r,c)) = min_v] ∧ [chain_inequality_max(cell(r,c)) = max_v]
                    → [candidates(cell(r,c)) := candidates(cell(r,c)) ∩ {min_v, ..., max_v}]
                """,
                "logic": "Use inequality chains to determine minimum and maximum possible values for cells",
                "complexity": "easy",
                "composite": True,
                "composed_of": ["constraint_propagation", "forced_by_inequality"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["inequality_chains"]
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
                "composite": True,
                "composed_of": ["row_uniqueness", "constraint_propagation"],
                "prerequisites": ["row_uniqueness"],
                "applies_to": ["rows"]
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
                "composite": True,
                "composed_of": ["column_uniqueness", "constraint_propagation"],
                "prerequisites": ["column_uniqueness"],
                "applies_to": ["columns"]
            },
            
            "direct_constraint_forcing": {
                "name": "Direct Constraint Forcing",
                "description": "When constraint directly forces a specific value",
                "fol_rule": """
                ∀cell(r1,c1) ∀cell(r2,c2) ∀constraint(op):
                    [assigned(cell(r1,c1), v1)] ∧ [inequality(cell(r1,c1), op, cell(r2,c2))]
                    ∧ [|{v ∈ candidates(cell(r2,c2)) : satisfies(v1, op, v)}| = 1]
                    → [assign(cell(r2,c2), unique_satisfying_value)]
                """,
                "logic": "When one cell is assigned and constraint allows only one value in adjacent cell",
                "complexity": "easy",
                "composite": True,
                "composed_of": ["constraint_propagation", "forced_by_inequality"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["constraint_pairs"]
            }
        }
    
    def get_strategy(self, strategy_name: str) -> Dict:
        """Get strategy details by name"""
        return self.strategies.get(strategy_name, {})
    
    def list_strategies(self) -> List[str]:
        """Get list of all strategy names"""
        return list(self.strategies.keys())
    
    def get_all_strategies(self) -> Dict:
        """Get all strategies"""
        return self.strategies
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            return strategy.get('description', f'Strategy: {strategy_name}')
        return f'Strategy: {strategy_name}'

    def get_strategy_patterns(self, strategy_name: str) -> List[str]:
        """Get applicable patterns for this strategy"""
        strategy = self.strategies.get(strategy_name, {})
        return strategy.get('applies_to', ['general'])

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name, {})
        return strategy.get('prerequisites', [])
    
    def get_composite_strategies(self) -> List[str]:
        """Get strategies that are composite (built from other strategies)"""
        return [name for name, strategy in self.strategies.items() 
                if strategy.get('composite', False)]
    
    def get_atomic_strategies(self) -> List[str]:
        """Get strategies that are atomic (not built from others)"""
        return [name for name, strategy in self.strategies.items() 
                if not strategy.get('composite', False)]
    
    def validate_compositionality(self) -> bool:
        """Validate that all composite strategies have valid prerequisites"""
        for name, strategy in self.strategies.items():
            if strategy.get('composite', False):
                prerequisites = strategy.get('prerequisites', [])
                for prereq in prerequisites:
                    if prereq not in self.strategies:
                        print(f"Error: Strategy '{name}' requires '{prereq}' which is not defined")
                        return False
        return True