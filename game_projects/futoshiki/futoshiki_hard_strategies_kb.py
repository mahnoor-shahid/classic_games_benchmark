# futoshiki_hard_strategies_kb.py
"""
Knowledge Base for Hard Futoshiki Solving Strategies
Contains FOL rules that compose easy and moderate strategies for advanced techniques
"""

from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from typing import List, Dict

class FutoshikiHardStrategiesKB:
    def __init__(self):
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.strategies = {
            "multiple_constraint_chains": {
                "name": "Multiple Constraint Chains",
                "description": "Analyze multiple intersecting inequality chains simultaneously",
                "fol_rule": """
                ∀chain_set{C1,C2,...,Cn} ∀intersection_cells{I1,I2,...,Im}:
                    [∀i ∈ {1,...,n}: constraint_chain(Ci)]
                    ∧ [∀j ∈ {1,...,m}: Ij ∈ intersection(C1,C2,...,Cn)]
                    ∧ [∀chain Ci: propagate_chain_constraints(Ci)]
                    ∧ [∀intersection Ij: resolve_chain_conflicts(Ij, affecting_chains)]
                    → [∀cell ∈ ⋃Ci: candidates(cell) := intersection_of_chain_constraints(cell, relevant_chains)]
                """,
                "logic": "Resolve conflicts between multiple intersecting constraint chains",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["constraint_chain_analysis", "constraint_propagation_advanced", "mutual_constraint_elimination"],
                "prerequisites": ["constraint_chain_analysis", "constraint_propagation_advanced"],
                "applies_to": ["complex_constraint_networks"],
                "constraint_aware": True
            },
            
            "constraint_network_analysis": {
                "name": "Constraint Network Analysis",
                "description": "Global analysis of entire constraint network for value forcing",
                "fol_rule": """
                ∀grid(G) ∀constraint_network(N) ∀value(v) ∀cell(r,c):
                    [complete_constraint_network(G, N)]
                    ∧ [∀assignment ∈ possible_assignments(cell(r,c), v): 
                       propagate_through_network(N, assignment) → contradiction ∨ forced_values]
                    ∧ [|{valid_assignments}| = 1]
                    → [assign(cell(r,c), unique_valid_value)]
                """,
                "logic": "Use global constraint network to determine forced assignments",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["multiple_constraint_chains", "value_forcing_by_uniqueness", "constraint_propagation_advanced"],
                "prerequisites": ["constraint_propagation_advanced", "constraint_chain_analysis"],
                "applies_to": ["global_constraints"],
                "constraint_aware": True
            },
            
            "naked_triple": {
                "name": "Naked Triple",
                "description": "Three cells with candidates forming a locked triple set",
                "fol_rule": """
                ∀unit(u) ∀cell_set{(r1,c1),(r2,c2),(r3,c3)} ∀value_set{v1,v2,v3}:
                    [∀i ∈ {1,2,3}: cell(ri,ci) ∈ unit(u)] ∧ [all_distinct((r1,c1),(r2,c2),(r3,c3))]
                    ∧ [⋃{candidates(cell(ri,ci)) : i ∈ {1,2,3}} = {v1,v2,v3}]
                    ∧ [∀i ∈ {1,2,3}: satisfy_all_adjacent_constraints(cell(ri,ci), candidates(cell(ri,ci)))]
                    → [∀cell(r',c') ∈ unit(u), (r',c') ∉ {(r1,c1),(r2,c2),(r3,c3)}: 
                       ∀v ∈ {v1,v2,v3}: remove_candidate(cell(r',c'), v)]
                """,
                "logic": "Three cells collectively containing only three values eliminate those from other cells",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["naked_pair", "constraint_propagation_advanced", "value_forcing_by_uniqueness"],
                "prerequisites": ["naked_pair", "constraint_propagation_advanced"],
                "applies_to": ["rows", "columns"],
                "constraint_aware": True
            },
            
            "hidden_triple": {
                "name": "Hidden Triple",
                "description": "Three values that can only go in three specific cells within a unit",
                "fol_rule": """
                ∀unit(u) ∀value_set{v1,v2,v3} ∀cell_set{(r1,c1),(r2,c2),(r3,c3)}:
                    [∀v ∈ {v1,v2,v3}: ∀cell(r,c) ∈ unit(u): 
                     v ∈ candidates(cell(r,c)) ∧ satisfy_constraints(cell(r,c), v) 
                     → (r,c) ∈ {(r1,c1),(r2,c2),(r3,c3)}]
                    → [∀i ∈ {1,2,3}: candidates(cell(ri,ci)) := candidates(cell(ri,ci)) ∩ {v1,v2,v3}]
                """,
                "logic": "If three values can only be placed in three cells, remove all other candidates from those cells",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["hidden_pair", "constraint_network_analysis", "value_forcing_by_uniqueness"],
                "prerequisites": ["hidden_pair", "value_forcing_by_uniqueness"],
                "applies_to": ["rows", "columns"],
                "constraint_aware": True
            },
            
            "constraint_intersection_forcing": {
                "name": "Constraint Intersection Forcing",
                "description": "Force values at intersections of multiple constraint chains",
                "fol_rule": """
                ∀cell(r,c) ∀constraint_chains{C1,C2,...,Cn}:
                    [∀i ∈ {1,...,n}: cell(r,c) ∈ constraint_chain(Ci)]
                    ∧ [∀i ∈ {1,...,n}: chain_constrains_cell(Ci, cell(r,c), valid_values_i)]
                    ∧ [intersection_values = ⋂{valid_values_i : i ∈ {1,...,n}}]
                    ∧ [|intersection_values| = 1]
                    → [assign(cell(r,c), unique_intersection_value)]
                """,
                "logic": "When multiple constraint chains intersect at a cell, use intersection of valid values",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["multiple_constraint_chains", "constraint_network_analysis"],
                "prerequisites": ["multiple_constraint_chains"],
                "applies_to": ["constraint_intersections"],
                "constraint_aware": True
            },
            
            "advanced_sandwich_analysis": {
                "name": "Advanced Sandwich Analysis",
                "description": "Complex analysis of cells constrained by multiple adjacent inequalities",
                "fol_rule": """
                ∀cell(r,c) ∀adjacent_constraints{(cell1,op1),(cell2,op2),...,(celln,opn)}:
                    [∀i ∈ {1,...,n}: adjacent(cell(r,c), celli) ∧ inequality(constraint_i)]
                    ∧ [∀value(v) ∈ candidates(cell(r,c)): 
                       check_all_adjacent_satisfiability(v, adjacent_constraints)]
                    ∧ [valid_values = {v : satisfies_all_adjacent(v, adjacent_constraints)}]
                    → [candidates(cell(r,c)) := valid_values]
                """,
                "logic": "Analyze cells with multiple adjacent constraints to determine valid value ranges",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["inequality_sandwich", "mutual_constraint_elimination", "constraint_splitting"],
                "prerequisites": ["inequality_sandwich", "mutual_constraint_elimination"],
                "applies_to": ["multi_constrained_cells"],
                "constraint_aware": True
            },
            
            "global_constraint_consistency": {
                "name": "Global Constraint Consistency",
                "description": "Ensure global consistency across all constraints simultaneously",
                "fol_rule": """
                ∀grid(G) ∀constraint_set(CS) ∀assignment_hypothesis(H):
                    [∀constraint(c) ∈ CS: local_consistency(c, H)]
                    ∧ [arc_consistency(CS, H)] ∧ [path_consistency(CS, H)]
                    ∧ [∃unique_solution: satisfies_all_constraints(unique_solution, CS)]
                    → [∀cell(r,c): candidates(cell(r,c)) := {v : consistent_with_global_solution(v, cell(r,c))}]
                """,
                "logic": "Use global constraint satisfaction to determine valid candidates",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["constraint_network_analysis", "advanced_sandwich_analysis", "constraint_intersection_forcing"],
                "prerequisites": ["constraint_network_analysis"],
                "applies_to": ["global_constraints"],
                "constraint_aware": True
            },
            
            "temporal_constraint_reasoning": {
                "name": "Temporal Constraint Reasoning",
                "description": "Reason about sequence of moves and their constraint implications",
                "fol_rule": """
                ∀move_sequence{m1,m2,...,mk} ∀constraint_state{s1,s2,...,sk}:
                    [∀i ∈ {1,...,k}: apply_move(mi, si-1) → si]
                    ∧ [∀i ∈ {1,...,k}: constraint_consistent(si)]
                    ∧ [leads_to_contradiction(move_sequence) ∨ leads_to_solution(move_sequence)]
                    → [eliminate_contradictory_paths() ∧ force_solution_paths()]
                """,
                "logic": "Use forward reasoning about move sequences to eliminate impossible candidates",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["global_constraint_consistency", "constraint_network_analysis", "naked_triple"],
                "prerequisites": ["global_constraint_consistency"],
                "applies_to": ["solution_paths"],
                "constraint_aware": True
            },
            
            "constraint_symmetry_breaking": {
                "name": "Constraint Symmetry Breaking",
                "description": "Use symmetry properties to break constraint solving search space",
                "fol_rule": """
                ∀grid(G) ∀symmetry_group(SG) ∀equivalent_states{s1,s2,...,sn}:
                    [∀si ∈ equivalent_states: constraint_equivalent(s1, si, SG)]
                    ∧ [∀si ∈ equivalent_states: solution_equivalent(s1, si)]
                    ∧ [select_canonical_representative(s1, equivalent_states)]
                    → [eliminate_non_canonical_assignments(equivalent_states \ {s1})]
                """,
                "logic": "Use problem symmetries to reduce search space and force unique solutions",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["temporal_constraint_reasoning", "global_constraint_consistency"],
                "prerequisites": ["global_constraint_consistency"],
                "applies_to": ["symmetric_patterns"],
                "constraint_aware": True
            },
            
            "constraint_forcing_networks": {
                "name": "Constraint Forcing Networks",
                "description": "Use network analysis to identify forced value assignments",
                "fol_rule": """
                ∀constraint_network(N) ∀cell(r,c) ∀value(v):
                    [build_forcing_network(N, cell(r,c), v)]
                    ∧ [∀path ∈ forcing_paths(N): leads_to_contradiction(path) ∨ leads_to_solution(path)]
                    ∧ [all_paths_consistent(forcing_paths(N)) → forced_assignment(cell(r,c), v)]
                    → [assign(cell(r,c), v)]
                """,
                "logic": "Build networks of constraint implications to identify forced assignments",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["constraint_intersection_forcing", "temporal_constraint_reasoning"],
                "prerequisites": ["constraint_intersection_forcing"],
                "applies_to": ["forcing_networks"],
                "constraint_aware": True
            },
            
            "advanced_constraint_propagation": {
                "name": "Advanced Constraint Propagation",
                "description": "Multi-level constraint propagation with conflict analysis",
                "fol_rule": """
                ∀grid(G) ∀constraint_set(CS) ∀propagation_levels{L1,L2,...,Ln}:
                    [∀level Li: apply_propagation_rules(Li, CS)]
                    ∧ [detect_conflicts(CS) → analyze_conflict_sources()]
                    ∧ [resolve_conflicts() → new_forced_assignments()]
                    ∧ [fixed_point_reached(CS)]
                    → [apply_all_deduced_assignments()]
                """,
                "logic": "Apply multi-level constraint propagation with intelligent conflict resolution",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["constraint_propagation_advanced", "constraint_forcing_networks"],
                "prerequisites": ["constraint_propagation_advanced"],
                "applies_to": ["global_propagation"],
                "constraint_aware": True
            },
            
            "bidirectional_constraint_chaining": {
                "name": "Bidirectional Constraint Chaining",
                "description": "Analyze constraint chains in both directions to find forced values",
                "fol_rule": """
                ∀constraint_chain(C) ∀direction{forward,backward} ∀cell(r,c):
                    [cell(r,c) ∈ constraint_chain(C)]
                    ∧ [forward_chain_analysis(C, cell(r,c)) → forward_constraints]
                    ∧ [backward_chain_analysis(C, cell(r,c)) → backward_constraints]
                    ∧ [intersection_constraints = forward_constraints ∩ backward_constraints]
                    → [candidates(cell(r,c)) := candidates(cell(r,c)) ∩ intersection_constraints]
                """,
                "logic": "Use bidirectional analysis of constraint chains to narrow candidate sets",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["constraint_chain_analysis", "multiple_constraint_chains"],
                "prerequisites": ["constraint_chain_analysis"],
                "applies_to": ["constraint_chains"],
                "constraint_aware": True
            },
            
            "constraint_conflict_resolution": {
                "name": "Constraint Conflict Resolution",
                "description": "Resolve conflicts between competing constraint implications",
                "fol_rule": """
                ∀conflict_set{c1,c2,...,cn} ∀resolution_strategy(R):
                    [∀i ∈ {1,...,n}: constraint_conflict(ci)]
                    ∧ [analyze_conflict_sources(conflict_set) → conflict_graph]
                    ∧ [apply_resolution_strategy(R, conflict_graph) → resolution_actions]
                    ∧ [validate_resolution(resolution_actions) → consistent_state]
                    → [apply_resolution_actions(resolution_actions)]
                """,
                "logic": "Systematically resolve conflicts between constraint implications",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["advanced_constraint_propagation", "global_constraint_consistency"],
                "prerequisites": ["advanced_constraint_propagation"],
                "applies_to": ["conflict_resolution"],
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
    
    def get_moderate_strategies(self) -> Dict:
        """Get moderate strategies from moderate KB"""
        return self.moderate_kb.get_all_strategies()
    
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
        all_available = (set(self.list_strategies()) | 
                        set(self.easy_kb.list_strategies()) | 
                        set(self.moderate_kb.list_strategies()))
        
        for name, strategy in self.strategies.items():
            if strategy.get('composite', False):
                prerequisites = strategy.get('prerequisites', [])
                for prereq in prerequisites:
                    if prereq not in all_available:
                        print(f"Error: Strategy '{name}' requires '{prereq}' which is not available")
                        return False
        return True
    
    def get_strategy_complexity_score(self, strategy_name: str) -> float:
        """Get numerical complexity score for a strategy"""
        complexity_scores = {
            'multiple_constraint_chains': 3.0,
            'constraint_network_analysis': 3.5,
            'naked_triple': 2.8,
            'hidden_triple': 3.2,
            'constraint_intersection_forcing': 3.4,
            'advanced_sandwich_analysis': 3.6,
            'global_constraint_consistency': 4.0,
            'temporal_constraint_reasoning': 4.5,
            'constraint_symmetry_breaking': 4.2,
            'constraint_forcing_networks': 3.8,
            'advanced_constraint_propagation': 3.3,
            'bidirectional_constraint_chaining': 3.7,
            'constraint_conflict_resolution': 4.3
        }
        return complexity_scores.get(strategy_name, 3.0)
    
    def get_strategies_by_complexity(self) -> Dict[str, List[str]]:
        """Get strategies grouped by complexity level"""
        by_complexity = {
            'basic_hard': [],
            'intermediate_hard': [],
            'advanced_hard': [],
            'expert_hard': []
        }
        
        for strategy_name in self.list_strategies():
            score = self.get_strategy_complexity_score(strategy_name)
            if score < 3.2:
                by_complexity['basic_hard'].append(strategy_name)
            elif score < 3.7:
                by_complexity['intermediate_hard'].append(strategy_name)
            elif score < 4.2:
                by_complexity['advanced_hard'].append(strategy_name)
            else:
                by_complexity['expert_hard'].append(strategy_name)
        
        return by_complexity