# kenken_moderate_strategies_kb.py
"""
Knowledge Base for Moderate Ken Ken Solving Strategies
Contains FOL rules that compose easy strategies for intermediate techniques
"""

from kenken_easy_strategies_kb import KenKenEasyStrategiesKB
from typing import List, Dict

class KenKenModerateStrategiesKB:
    def __init__(self):
        self.easy_kb = KenKenEasyStrategiesKB()
        self.strategies = {
            "cage_elimination": {
                "name": "Cage Elimination",
                "description": "Eliminate candidates based on partial cage assignments and remaining target",
                "fol_rule": """
                ∀cage(c) ∀assigned_cells(A) ∀empty_cells(E) ∀current_sum(s) ∀target(t):
                    [cage_operation(c) = add] ∧ [A ∪ E = cells_in_cage(c)] ∧ [A ∩ E = ∅]
                    ∧ [sum(values(A)) = s] ∧ [cage_target(c) = t]
                    → [∀cell ∈ E: ∀v ∈ candidates(cell): 
                       if impossible_to_complete_cage(E, v, t-s) then remove_candidate(cell, v)]
                """,
                "logic": "Remove candidates that would make it impossible to complete the cage target",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_completion", "simple_addition_cage", "naked_single"]
            },
            
            "complex_arithmetic_cages": {
                "name": "Complex Arithmetic Cages",
                "description": "Solve larger cages with multiple operations and constraints",
                "fol_rule": """
                ∀cage(c) ∀operation(op) ∀target(t) ∀cells(C) ∀grid_size(n):
                    [cage_operation(c) = op] ∧ [cage_target(c) = t] ∧ [|C| ≥ 3]
                    ∧ [∀cell ∈ C: assigned(cell) ∨ |candidates(cell)| > 1]
                    → [apply_constraint_propagation(c, op, t, C, n)]
                """,
                "logic": "Use constraint propagation for complex cages with multiple cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_elimination", "basic_multiplication_cage", "simple_addition_cage"]
            },
            
            "multi_cage_analysis": {
                "name": "Multi-Cage Analysis",
                "description": "Analyze interactions between overlapping or adjacent cages",
                "fol_rule": """
                ∀cages{c1,c2} ∀shared_cells(S) ∀values(V):
                    [cells_in_cage(c1) ∩ cells_in_cage(c2) = S] ∧ [|S| ≥ 1]
                    ∧ [∀cell ∈ S: possible_values_for_cage(cell, c1) ∩ possible_values_for_cage(cell, c2) = V]
                    → [∀cell ∈ S: candidates(cell) := candidates(cell) ∩ V]
                """,
                "logic": "Constrain shared cells based on requirements from multiple cages",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_elimination", "hidden_single_row", "hidden_single_column"]
            },
            
            "advanced_factorization": {
                "name": "Advanced Factorization",
                "description": "Use prime factorization and divisibility rules for multiplication/division cages",
                "fol_rule": """
                ∀cage(c) ∀operation(multiply) ∀target(t) ∀cells(C) ∀prime_factors(P):
                    [cage_operation(c) = multiply] ∧ [cage_target(c) = t] 
                    ∧ [prime_factorization(t) = P] ∧ [|C| = |P|]
                    → [if unique_assignment_exists(P, C, grid_constraints) 
                       then assign_prime_factors(C, P)]
                """,
                "logic": "Use prime factorization to solve multiplication cages when factors are constrained",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["basic_multiplication_cage", "basic_division_cage", "naked_single"]
            },
            
            "constraint_propagation": {
                "name": "Constraint Propagation",
                "description": "Propagate constraints through the grid using cage and Latin square rules",
                "fol_rule": """
                ∀cell(r,c) ∀value(v) ∀affected_cells(A):
                    [assign(cell(r,c), v)] ∧ [A = cells_affected_by_assignment(r,c,v)]
                    → [∀cell' ∈ A: propagate_constraints(cell', v, cage_constraints, latin_square_constraints)]
                """,
                "logic": "When assigning a value, propagate all resulting constraints through the grid",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["eliminate_by_row", "eliminate_by_column", "cage_elimination"]
            },
            
            "partial_sum_analysis": {
                "name": "Partial Sum Analysis",
                "description": "Analyze partial sums in addition cages to constrain remaining cells",
                "fol_rule": """
                ∀cage(c) ∀assigned_subset(A) ∀remaining_cells(R) ∀partial_sum(ps) ∀target(t):
                    [cage_operation(c) = add] ∧ [A ⊂ cells_in_cage(c)] ∧ [R = cells_in_cage(c) \ A]
                    ∧ [sum(values(A)) = ps] ∧ [cage_target(c) = t] ∧ [remaining_target = t - ps]
                    → [constrain_cells_by_remaining_sum(R, remaining_target, |R|)]
                """,
                "logic": "Use partial sums to constrain the possible values in remaining cage cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["simple_addition_cage", "cage_completion", "constraint_propagation"]
            },
            
            "quotient_remainder_analysis": {
                "name": "Quotient-Remainder Analysis",
                "description": "Use division properties to constrain values in division cages",
                "fol_rule": """
                ∀cage(c) ∀operation(divide) ∀target(t) ∀dividend(d) ∀divisor(s):
                    [cage_operation(c) = divide] ∧ [cage_target(c) = t]
                    ∧ [dividend_cell ∈ cells_in_cage(c)] ∧ [divisor_cell ∈ cells_in_cage(c)]
                    ∧ [d = value(dividend_cell)] ∧ [s = value(divisor_cell)]
                    → [d = t × s] ∧ [constrain_by_divisibility(d, s, t)]
                """,
                "logic": "Use divisibility constraints to solve division cages",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["basic_division_cage", "advanced_factorization"]
            },
            
            "cage_intersection": {
                "name": "Cage Intersection",
                "description": "Analyze cells that belong to multiple constraint regions",
                "fol_rule": """
                ∀cell(r,c) ∀cages{c1,c2,...,cn} ∀constraints{con1,con2,...,conn}:
                    [∀i ∈ {1,...,n}: cell(r,c) ∈ cage(ci)] ∧ [∀i: constraint(ci) = coni]
                    → [candidates(cell(r,c)) := ⋂{values_satisfying_constraint(coni) : i ∈ {1,...,n}}]
                """,
                "logic": "Cells in multiple cages must satisfy all cage constraints simultaneously",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["multi_cage_analysis", "constraint_propagation"]
            },
            
            "elimination_by_exhaustion": {
                "name": "Elimination by Exhaustion",
                "description": "Try all possibilities for a cell and eliminate those that lead to contradictions",
                "fol_rule": """
                ∀cell(r,c) ∀candidates{v1,v2,...,vk} ∀grid_state(G):
                    [candidates(cell(r,c)) = {v1,v2,...,vk}] ∧ [k ≤ 3]
                    → [∀vi: if assignment_leads_to_contradiction(cell(r,c), vi, G) 
                       then remove_candidate(cell(r,c), vi)]
                """,
                "logic": "Remove candidates that lead to contradictions when tested",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["constraint_propagation", "cage_elimination", "naked_single"]
            },
            
            "symmetric_cage_analysis": {
                "name": "Symmetric Cage Analysis", 
                "description": "Use symmetry properties in cage arrangements to deduce values",
                "fol_rule": """
                ∀cages{c1,c2} ∀symmetry_relation(R):
                    [symmetric_cages(c1, c2, R)] ∧ [cage_operation(c1) = cage_operation(c2)]
                    ∧ [cage_target(c1) = cage_target(c2)] ∧ [|cells_in_cage(c1)| = |cells_in_cage(c2)|]
                    → [apply_symmetry_constraints(c1, c2, R)]
                """,
                "logic": "Use symmetrical cage properties to constrain solutions",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["multi_cage_analysis", "constraint_propagation"]
            }
        }
    
    def get_strategy(self, strategy_name):
        return self.strategies.get(strategy_name)
    
    def list_strategies(self):
        return list(self.strategies.keys())
    
    def get_all_strategies(self):
        return self.strategies
    
    def get_easy_strategies(self):
        return self.easy_kb.get_all_strategies()
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            return strategy.get('description', f'Strategy: {strategy_name}')
        return f'Strategy: {strategy_name}'

    def get_strategy_patterns(self, strategy_name: str) -> List[str]:
        """Get applicable patterns for this strategy"""
        if 'cage' in strategy_name:
            return ['cage', 'arithmetic', 'constraint']
        elif 'elimination' in strategy_name:
            return ['elimination', 'constraint']
        elif 'factorization' in strategy_name:
            return ['factorization', 'multiplication', 'prime']
        elif 'propagation' in strategy_name:
            return ['propagation', 'constraint', 'cascade']
        elif 'analysis' in strategy_name:
            return ['analysis', 'deduction', 'complex']
        else:
            return ['moderate', 'intermediate']

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy and strategy.get('composite', False):
            return strategy.get('composed_of', [])
        
        # For non-composite strategies, return basic prerequisites
        basic_prereqs = ['naked_single', 'single_cell_cage']
        if strategy_name not in basic_prereqs:
            return basic_prereqs
        return []
    
    def get_operations_used(self, strategy_name: str) -> List[str]:
        """Get arithmetic operations used by this strategy"""
        if 'addition' in strategy_name or 'sum' in strategy_name:
            return ['add']
        elif 'subtraction' in strategy_name:
            return ['subtract']
        elif 'multiplication' in strategy_name or 'factorization' in strategy_name:
            return ['multiply']
        elif 'division' in strategy_name or 'quotient' in strategy_name:
            return ['divide']
        elif 'arithmetic' in strategy_name:
            return ['add', 'subtract', 'multiply', 'divide']
        else:
            return []