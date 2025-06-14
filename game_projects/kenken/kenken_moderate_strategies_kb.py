# kenken_moderate_strategies_kb.py
"""
Knowledge Base for Moderate KenKen Solving Strategies
Contains FOL rules that compose easy strategies for intermediate techniques
"""

from kenken_easy_strategies_kb import EasyKenKenStrategiesKB

class ModerateKenKenStrategiesKB:
    def __init__(self):
        self.easy_kb = EasyKenKenStrategiesKB()
        self.strategies = {
            "cage_candidate_elimination": {
                "name": "Cage Candidate Elimination",
                "description": "Eliminate candidates by analyzing cage constraints with row/column conflicts",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀value(v):
                    [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    ∧ [same_row(r1,r2) ∨ same_column(c1,c2)]
                    ∧ [v ∈ candidates(cell(r1,c1))] ∧ [v ∈ candidates(cell(r2,c2))]
                    ∧ [¬∃valid_cage_assignment: contains(v, cell(r1,c1)) ∧ contains(v, cell(r2,c2))]
                    → [remove_candidate(cell(r1,c1), v) ∨ remove_candidate(cell(r2,c2), v)]
                """,
                "logic": "If placing same value in two cage cells violates row/column uniqueness, eliminate candidates",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["eliminate_by_row", "eliminate_by_column", "simple_cage_arithmetic"]
            },
            
            "multi_cage_intersection": {
                "name": "Multi-Cage Intersection Analysis",
                "description": "Analyze intersections between multiple cages to eliminate candidates",
                "fol_rule": """
                ∀cage(c1) ∀cage(c2) ∀cell(r,col) ∀value(v):
                    [cell(r,col) ∈ row_or_column_intersection(c1, c2)]
                    ∧ [v ∈ candidates(cell(r,col))]
                    ∧ [assignment_to_cage(c1, v) conflicts_with assignment_to_cage(c2, v)]
                    → [remove_candidate(cell(r,col), v)]
                """,
                "logic": "Use intersections between cages to eliminate candidates that create conflicts",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_boundary_constraint", "eliminate_by_row", "eliminate_by_column"]
            },
            
            "cage_combination_analysis": {
                "name": "Cage Combination Analysis",
                "description": "Analyze combinations of values that satisfy multiple adjacent cages",
                "fol_rule": """
                ∀cage_set{c1,c2,...,cn} ∀shared_region(R):
                    [∀ci ∈ cage_set: intersects(ci, R)]
                    ∧ [limited_value_combinations(cage_set, R)]
                    → [∀cell(r,col) ∈ R: restrict_candidates_to_valid_combinations(cell(r,col))]
                """,
                "logic": "When multiple cages share cells/constraints, limit candidates to valid combinations",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["two_cell_addition_cage", "two_cell_subtraction_cage", "cage_boundary_constraint"]
            },
            
            "advanced_cage_arithmetic": {
                "name": "Advanced Cage Arithmetic",
                "description": "Use advanced arithmetic reasoning for larger cages",
                "fol_rule": """
                ∀cage(c) ∀target(t) ∀operation(op) ∀partial_assignment(P):
                    [cage_size(c) ≥ 3] ∧ [cage_target(c) = t] ∧ [cage_operation(c) = op]
                    ∧ [some_cells_assigned(cage(c), P)]
                    → [∀unassigned_cell ∈ cage(c): 
                       candidates(unassigned_cell) = valid_completions(P, t, op, cage(c))]
                """,
                "logic": "For larger cages with partial assignments, calculate valid completions",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["simple_cage_arithmetic", "cage_completion", "naked_single"]
            },
            
            "cage_sum_distribution": {
                "name": "Cage Sum Distribution",
                "description": "Distribute large sums across cage cells considering row/column constraints",
                "fol_rule": """
                ∀cage(c) ∀addition_target(t) ∀grid_size(n):
                    [cage_operation(c) = addition] ∧ [cage_target(c) = t] ∧ [cage_size(c) ≥ 3]
                    ∧ [t > cage_size(c) × n/2]  // Large sum constraint
                    → [∀cell(r,col) ∈ cage(c): 
                       candidates(cell(r,col)) ⊆ {v : v ≥ minimum_required_for_large_sum(t, cage_size(c), n)}]
                """,
                "logic": "For large sum cages, cells must contain relatively large values",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["two_cell_addition_cage", "hidden_single_row", "hidden_single_column"]
            },
            
            "cage_product_factorization": {
                "name": "Cage Product Factorization",
                "description": "Factor multiplication targets to find valid number combinations",
                "fol_rule": """
                ∀cage(c) ∀multiplication_target(t) ∀grid_size(n):
                    [cage_operation(c) = multiplication] ∧ [cage_target(c) = t]
                    ∧ [factorizations(t) = F] ∧ [valid_in_grid_size(F, n)]
                    → [∀cell(r,col) ∈ cage(c): 
                       candidates(cell(r,col)) ⊆ ⋃{factors(f) : f ∈ F ∧ compatible_with_cage_size(f, cage(c))}]
                """,
                "logic": "Use prime factorization to limit candidates in multiplication cages",
                "complexity": "moderate", 
                "composite": True,
                "composed_of": ["two_cell_multiplication_cage", "forced_candidate_cage", "eliminate_by_row"]
            },
            
            "naked_pair_in_cage": {
                "name": "Naked Pair in Cage",
                "description": "Find pairs of cells in cages with identical candidate sets",
                "fol_rule": """
                ∀cage(c) ∀cell(r1,c1) ∀cell(r2,c2) ∀value_set{v1,v2}:
                    [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)] ∧ [(r1,c1) ≠ (r2,c2)]
                    ∧ [candidates(cell(r1,c1)) = {v1,v2}] ∧ [candidates(cell(r2,c2)) = {v1,v2}]
                    ∧ [same_row(r1,r2) ∨ same_column(c1,c2)]
                    → [∀cell(r',c') ∈ same_line(r1,c1,r2,c2): 
                       cell(r',c') ≠ cell(r1,c1) ∧ cell(r',c') ≠ cell(r2,c2) 
                       → remove_candidate(cell(r',c'), v1) ∧ remove_candidate(cell(r',c'), v2)]
                """,
                "logic": "If two cells in same row/column have identical two-candidate sets, eliminate those values elsewhere",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["naked_single", "eliminate_by_row", "eliminate_by_column"]
            },
            
            "hidden_pair_in_cage": {
                "name": "Hidden Pair in Cage",
                "description": "Find pairs of values that can only go in two specific cells",
                "fol_rule": """
                ∀region(R) ∀value_set{v1,v2} ∀cell_set{(r1,c1),(r2,c2)}:
                    [region(R) = row(r) ∨ region(R) = column(c)]
                    ∧ [∀v ∈ {v1,v2}: ∀cell(r,c) ∈ region(R): 
                       v ∈ candidates(cell(r,c)) → (r,c) ∈ {(r1,c1),(r2,c2)}]
                    → [∀i ∈ {1,2}: candidates(cell(ri,ci)) := candidates(cell(ri,ci)) ∩ {v1,v2}]
                """,
                "logic": "If two values can only be placed in two cells, remove all other candidates from those cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_single_row", "hidden_single_column", "cage_boundary_constraint"]
            },
            
            "cage_constraint_propagation": {
                "name": "Cage Constraint Propagation",
                "description": "Propagate constraints from completed cages to adjacent regions",
                "fol_rule": """
                ∀cage(c1) ∀cage(c2) ∀shared_line(L):
                    [cage_partially_solved(c1)] ∧ [shares_line(c1, c2, L)]
                    ∧ [known_values_in_cage(c1) = V]
                    → [∀cell(r,col) ∈ intersection(c2, L): 
                       ∀v ∈ V: remove_candidate(cell(r,col), v)]
                """,
                "logic": "When cage values are determined, propagate exclusions to intersecting cages",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_completion", "eliminate_by_row", "eliminate_by_column"]
            },
            
            "division_remainder_analysis": {
                "name": "Division Remainder Analysis",
                "description": "Use division remainders to constrain candidate values",
                "fol_rule": """
                ∀cage(c) ∀division_target(t) ∀grid_size(n):
                    [cage_operation(c) = division] ∧ [cage_target(c) = t] ∧ [cage_size(c) = 2]
                    ∧ [cell(r1,c1) ∈ cage(c)] ∧ [cell(r2,c2) ∈ cage(c)]
                    → [candidates(cell(r1,c1)) = {v1 : ∃v2 ∈ [1,n], (v1/v2 = t ∨ v2/v1 = t) 
                       ∧ integer_division(v1,v2) ∧ valid_in_context(v1,v2)}]
                """,
                "logic": "For division cages, ensure quotients are exact integers with valid grid constraints",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["two_cell_division_cage", "cage_boundary_constraint", "naked_single"]
            },
            
            "large_cage_symmetry": {
                "name": "Large Cage Symmetry",
                "description": "Use symmetry patterns in large cages to eliminate candidates",
                "fol_rule": """
                ∀cage(c) ∀symmetry_pattern(P):
                    [cage_size(c) ≥ 4] ∧ [has_symmetry(cage(c), P)]
                    ∧ [symmetric_positions(cage(c)) = SP]
                    → [∀(pos1, pos2) ∈ SP: 
                       candidates(pos1) must_be_compatible_with candidates(pos2)]
                """,
                "logic": "In symmetric cage arrangements, corresponding positions must have compatible values",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["advanced_cage_arithmetic", "multi_cage_intersection", "naked_pair_in_cage"]
            },
            
            "cage_endpoint_analysis": {
                "name": "Cage Endpoint Analysis",
                "description": "Analyze extreme values (min/max) that can appear in cage endpoints",
                "fol_rule": """
                ∀cage(c) ∀operation(op) ∀target(t) ∀grid_size(n):
                    [cage_operation(c) = op] ∧ [cage_target(c) = t]
                    ∧ [extreme_value_constraints(op, t, n) = EC]
                    → [∀cell(r,col) ∈ cage(c): 
                       candidates(cell(r,col)) ⊆ feasible_range(EC, cage_size(c))]
                """,
                "logic": "Use min/max analysis to constrain possible values in cage cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["cage_sum_distribution", "cage_product_factorization", "forced_candidate_cage"]
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