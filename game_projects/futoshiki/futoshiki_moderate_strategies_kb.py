# futoshiki_moderate_strategies_kb.py
"""
Knowledge Base for Moderate Futoshiki Solving Strategies
Contains FOL rules that compose easy strategies for intermediate techniques
"""

from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from typing import List, Dict

class FutoshikiModerateStrategiesKB:
    def __init__(self):
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.strategies = {
            "naked_pair": {
                "name": "Naked Pair",
                "description": "Two cells in same unit have identical two-candidate sets",
                "fol_rule": """
                ∀unit(u) ∀cell(r1,c1) ∀cell(r2,c2) ∀value_set{v1,v2}:
                    [cell(r1,c1) ∈ unit(u)] ∧ [cell(r2,c2) ∈ unit(u)] ∧ [(r1,c1) ≠ (r2,c2)]
                    ∧ [candidates(cell(r1,c1)) = {v1,v2}] ∧ [candidates(cell(r2,c2)) = {v1,v2}]
                    ∧ [satisfy_all_constraints(cell(r1,c1), {v1,v2})] ∧ [satisfy_all_constraints(cell(r2,c2), {v1,v2})]
                    → [∀cell(r',c') ∈ unit(u), (r',c') ∉ {(r1,c1),(r2,c2)}: 
                       remove_candidate(cell(r',c'), v1) ∧ remove_candidate(cell(r',c'), v2)]
                """,
                "logic": "Two cells with identical candidates eliminate those values from other cells in the unit",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["row_uniqueness", "column_uniqueness", "constraint_propagation"],
                "prerequisites": ["constraint_propagation", "row_uniqueness", "column_uniqueness"],
                "applies_to": ["rows", "columns"],
                "constraint_aware": True
            },
            
            "hidden_pair": {
                "name": "Hidden Pair",
                "description": "Two values that can only go in two specific cells within a unit",
                "fol_rule": """
                ∀unit(u) ∀value_set{v1,v2} ∀cell_set{(r1,c1),(r2,c2)}:
                    [∀v ∈ {v1,v2}: ∀cell(r,c) ∈ unit(u): 
                     v ∈ candidates(cell(r,c)) ∧ satisfy_constraints(cell(r,c), v) → (r,c) ∈ {(r1,c1),(r2,c2)}]
                    → [∀i ∈ {1,2}: candidates(cell(ri,ci)) := candidates(cell(ri,ci)) ∩ {v1,v2}]
                """,
                "logic": "If two values can only be placed in two cells, remove all other candidates from those cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_single_row", "hidden_single_column", "constraint_propagation"],
                "prerequisites": ["hidden_single_row", "hidden_single_column"],
                "applies_to": ["rows", "columns"],
                "constraint_aware": True
            },
            
            "constraint_chain_analysis": {
                "name": "Constraint Chain Analysis",
                "description": "Analyze chains of inequalities to determine value ranges",
                "fol_rule": """
                ∀cell_chain{c1,c2,...,cn} ∀constraint_chain{op1,op2,...,op(n-1)}:
                    [∀i ∈ {1,...,n-1}: inequality(ci, opi, c(i+1))]
                    ∧ [chain_consistent(c1,...,cn, op1,...,op(n-1))]
                    → [∀i ∈ {1,...,n}: candidates(ci) := candidates(ci) ∩ valid_range_in_chain(ci, chain)]
                """,
                "logic": "Use inequality chains to constrain possible values based on transitivity",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["minimum_maximum_bounds", "constraint_propagation", "forced_by_inequality"],
                "prerequisites": ["minimum_maximum_bounds", "constraint_propagation"],
                "applies_to": ["inequality_chains"],
                "constraint_aware": True
            },
            
            "constraint_splitting": {
                "name": "Constraint Splitting",
                "description": "Split constraints based on possible value assignments",
                "fol_rule": """
                ∀cell(r1,c1) ∀cell(r2,c2) ∀constraint(op) ∀value_partition{P1,P2}:
                    [inequality(cell(r1,c1), op, cell(r2,c2))]
                    ∧ [candidates(cell(r1,c1)) = P1 ∪ P2] ∧ [P1 ∩ P2 = ∅]
                    ∧ [∀v1 ∈ P1: ∀v2 ∈ candidates(cell(r2,c2)): ¬satisfies(v1, op, v2)]
                    → [candidates(cell(r1,c1)) := P2]
                """,
                "logic": "Eliminate candidate sets that cannot satisfy constraints with any adjacent value",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["constraint_propagation", "direct_constraint_forcing"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["constraint_pairs"],
                "constraint_aware": True
            },
            
            "mutual_constraint_elimination": {
                "name": "Mutual Constraint Elimination",
                "description": "Two cells mutually constrain each other's possible values",
                "fol_rule": """
                ∀cell(r1,c1) ∀cell(r2,c2) ∀constraint(op1,op2):
                    [inequality(cell(r1,c1), op1, cell(r2,c2))] ∧ [inequality(cell(r2,c2), op2, cell(r1,c1))]
                    ∧ [mutually_exclusive_constraints(op1, op2)]
                    → [∀v1 ∈ candidates(cell(r1,c1)), v2 ∈ candidates(cell(r2,c2)):
                       ¬(satisfies(v1, op1, v2) ∧ satisfies(v2, op2, v1)) 
                       → remove_candidate_pair(cell(r1,c1), v1, cell(r2,c2), v2)]
                """,
                "logic": "When cells have mutual constraints, eliminate incompatible value pairs",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["constraint_propagation", "constraint_splitting"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["constraint_pairs"],
                "constraint_aware": True
            },
            
            "inequality_sandwich": {
                "name": "Inequality Sandwich",
                "description": "Cell constrained by inequalities on both sides",
                "fol_rule": """
                ∀cell(r,c) ∀cell_left(r,c-1) ∀cell_right(r,c+1) ∀constraints(op1,op2):
                    [inequality(cell_left(r,c-1), op1, cell(r,c))] ∧ [inequality(cell(r,c), op2, cell_right(r,c+1))]
                    ∧ [assigned(cell_left(r,c-1), v1)] ∧ [assigned(cell_right(r,c+1), v3)]
                    → [candidates(cell(r,c)) := {v : satisfies(v1, op1, v) ∧ satisfies(v, op2, v3)}]
                """,
                "logic": "Cell between two assigned cells with constraints must satisfy both inequalities",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["direct_constraint_forcing", "constraint_chain_analysis"],
                "prerequisites": ["direct_constraint_forcing"],
                "applies_to": ["horizontal_triplets", "vertical_triplets"],
                "constraint_aware": True
            },
            
            "constraint_propagation_advanced": {
                "name": "Advanced Constraint Propagation",
                "description": "Multi-step constraint propagation with value elimination",
                "fol_rule": """
                ∀cell(r,c) ∀constraint_network(N):
                    [cell(r,c) ∈ constraint_network(N)]
                    ∧ [∀constraint ∈ N: propagate_constraint(constraint) → new_eliminations]
                    ∧ [fixed_point_reached(N)]
                    → [apply_all_eliminations(N)]
                """,
                "logic": "Iteratively propagate constraints until no more eliminations possible",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["constraint_propagation", "mutual_constraint_elimination", "constraint_chain_analysis"],
                "prerequisites": ["constraint_propagation"],
                "applies_to": ["global_constraints"],
                "constraint_aware": True
            },
            
            "value_forcing_by_uniqueness": {
                "name": "Value Forcing by Uniqueness",
                "description": "Force values based on uniqueness constraints combined with inequalities",
                "fol_rule": """
                ∀unit(u) ∀value(v) ∀cell(r,c):
                    [cell(r,c) ∈ unit(u)] ∧ [v ∈ candidates(cell(r,c))]
                    ∧ [∀cell'(r',c') ∈ unit(u), (r',c') ≠ (r,c): 
                       v ∉ candidates(cell'(r',c')) ∨ ¬satisfy_adjacent_constraints(cell'(r',c'), v)]
                    → [assign(cell(r,c), v)]
                """,
                "logic": "If a value can only be placed in one cell considering both uniqueness and constraints",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_single_row", "hidden_single_column", "constraint_propagation_advanced"],
                "prerequisites": ["hidden_single_row", "hidden_single_column"],
                "applies_to": ["rows", "columns"],
                "constraint_aware": True
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
    
    def get_easy_strategies(self) -> Dict:
        """Get easy strategies from easy KB"""
        return self.easy_kb.get_all_strategies()
    
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
    
    def get_constraint_aware_strategies(self) -> List[str]:
        """Get strategies that are specifically constraint-aware"""
        return [name for name, strategy in self.strategies.items() 
                if strategy.get('constraint_aware', False)]
    
    def validate_compositionality(self) -> bool:
        """Validate that all composite strategies have valid prerequisites"""
        all_available = set(self.list_strategies()) | set(self.easy_kb.list_strategies())
        
        for name, strategy in self.strategies.items():
            if strategy.get('composite', False):
                prerequisites = strategy.get('prerequisites', [])
                for prereq in prerequisites:
                    if prereq not in all_available:
                        print(f"Error: Strategy '{name}' requires '{prereq}' which is not available")
                        return False
        return True