# hard_strategies_kb.py
"""
Knowledge Base for Hard Sudoku Solving Strategies
Contains FOL rules that compose easy and moderate strategies for advanced techniques
"""

from sudoku_easy_strategies_kb import EasyStrategiesKB
from sudoku_moderate_strategies_kb import ModerateStrategiesKB
from typing import List, Dict

class HardStrategiesKB:
    def __init__(self):
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.strategies = {
            "swordfish": {
                "name": "Swordfish",
                "description": "Extension of X-Wing to three rows/columns",
                "fol_rule": """
                ∀rows{r1,r2,r3} ∀columns{c1,c2,c3} ∀value(v):
                    [all_distinct(r1,r2,r3)] ∧ [all_distinct(c1,c2,c3)]
                    ∧ [∀r ∈ {r1,r2,r3}: ∀c ∉ {c1,c2,c3}: v ∉ candidates(cell(r,c))]
                    ∧ [∀r ∈ {r1,r2,r3}: |{c ∈ {c1,c2,c3} : v ∈ candidates(cell(r,c))}| ≥ 2]
                    → [∀r ∉ {r1,r2,r3}: ∀c ∈ {c1,c2,c3}: remove_candidate(cell(r,c), v)]
                """,
                "logic": "Three-row/column extension of X-Wing for eliminating candidates",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["x_wing", "pointing_pairs", "box_line_reduction"]
            },
            
            "jellyfish": {
                "name": "Jellyfish",
                "description": "Extension of Swordfish to four rows/columns",
                "fol_rule": """
                ∀rows{r1,r2,r3,r4} ∀columns{c1,c2,c3,c4} ∀value(v):
                    [all_distinct(r1,r2,r3,r4)] ∧ [all_distinct(c1,c2,c3,c4)]
                    ∧ [∀r ∈ {r1,r2,r3,r4}: ∀c ∉ {c1,c2,c3,c4}: v ∉ candidates(cell(r,c))]
                    ∧ [∀r ∈ {r1,r2,r3,r4}: |{c ∈ {c1,c2,c3,c4} : v ∈ candidates(cell(r,c))}| ≥ 2]
                    → [∀r ∉ {r1,r2,r3,r4}: ∀c ∈ {c1,c2,c3,c4}: remove_candidate(cell(r,c), v)]
                """,
                "logic": "Four-row/column extension of Swordfish for eliminating candidates",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["swordfish", "x_wing", "pointing_pairs"]
            },
            
            "xyz_wing": {
                "name": "XYZ-Wing",
                "description": "Extension of XY-Wing with three candidates in pivot cell",
                "fol_rule": """
                ∀cell_pivot(rp,cp) ∀cell_wing1(r1,c1) ∀cell_wing2(r2,c2) ∀values{x,y,z}:
                    [candidates(cell_pivot(rp,cp)) = {x,y,z}] 
                    ∧ [candidates(cell_wing1(r1,c1)) = {x,z}] 
                    ∧ [candidates(cell_wing2(r2,c2)) = {y,z}]
                    ∧ [sees(cell_pivot(rp,cp), cell_wing1(r1,c1))] 
                    ∧ [sees(cell_pivot(rp,cp), cell_wing2(r2,c2))]
                    → [∀cell(r,c): sees(cell_pivot(rp,cp), cell(r,c)) 
                       ∧ sees(cell_wing1(r1,c1), cell(r,c)) ∧ sees(cell_wing2(r2,c2), cell(r,c)) 
                       → remove_candidate(cell(r,c), z)]
                """,
                "logic": "Extension of XY-Wing where pivot has three candidates",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["xy_wing", "naked_triple"]
            },
            
            "wxyz_wing": {
                "name": "WXYZ-Wing",
                "description": "Four-cell wing pattern with specific candidate constraints",
                "fol_rule": """
                ∀cell_pivot(rp,cp) ∀cells{(r1,c1),(r2,c2),(r3,c3)} ∀values{w,x,y,z}:
                    [candidates(cell_pivot(rp,cp)) ⊆ {w,x,y,z}] 
                    ∧ [|candidates(cell_pivot(rp,cp))| ≥ 2]
                    ∧ [∀i ∈ {1,2,3}: candidates(cell(ri,ci)) ⊆ {w,x,y,z} ∧ |candidates(cell(ri,ci))| = 2]
                    ∧ [∀i ∈ {1,2,3}: sees(cell_pivot(rp,cp), cell(ri,ci))]
                    ∧ [⋃{candidates(cell(ri,ci)) : i ∈ {1,2,3}} ∪ candidates(cell_pivot(rp,cp)) = {w,x,y,z}]
                    → [∀cell(r,c): (∀i ∈ {1,2,3}: sees(cell(ri,ci), cell(r,c))) 
                       → remove_candidate(cell(r,c), z)]
                """,
                "logic": "Four-cell wing pattern for advanced eliminations",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["xyz_wing", "xy_wing", "naked_triple"]
            },
            
            "als_xz": {
                "name": "Almost Locked Set XZ Rule",
                "description": "Two Almost Locked Sets connected by restricted common candidates",
                "fol_rule": """
                ∀ALS1(A1) ∀ALS2(A2) ∀values{x,z}:
                    [almost_locked_set(A1)] ∧ [almost_locked_set(A2)]
                    ∧ [x ∈ candidates(A1) ∩ candidates(A2)]
                    ∧ [z ∈ candidates(A1) ∩ candidates(A2)]
                    ∧ [x ≠ z] ∧ [¬share_unit(A1, A2)]
                    ∧ [restricted_common_candidates(A1, A2, {x,z})]
                    → [∀cell(r,c): sees_ALS(cell(r,c), A1, z) ∧ sees_ALS(cell(r,c), A2, z) 
                       → remove_candidate(cell(r,c), z)]
                """,
                "logic": "Use Almost Locked Sets to eliminate candidates through XZ rule",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["naked_triple", "hidden_triple", "simple_coloring"]
            },
            
            "sue_de_coq": {
                "name": "Sue de Coq",
                "description": "Complex pattern involving box-line interactions with specific constraints",
                "fol_rule": """
                ∀box(b) ∀line(l) ∀cell_set(S) ∀value_sets{V1,V2,V3}:
                    [intersection(box(b), line(l)) = S] ∧ [|S| ≥ 2]
                    ∧ [candidates(S) = V1] ∧ [|V1| = |S| + 1]
                    ∧ [other_cells_in_box(b, S) has candidates V2]
                    ∧ [other_cells_in_line(l, S) has candidates V3]
                    ∧ [V2 ∩ V3 = ∅] ∧ [V1 = V2 ∪ V3]
                    → [eliminate V2 from line(l) ∖ S] ∧ [eliminate V3 from box(b) ∖ S]
                """,
                "logic": "Complex box-line interaction eliminating candidates through partitioned sets",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["box_line_reduction", "pointing_pairs", "naked_triple"]
            },
            
            "death_blossom": {
                "name": "Death Blossom",
                "description": "Advanced pattern with stem cell connected to multiple Almost Locked Sets",
                "fol_rule": """
                ∀cell_stem(rs,cs) ∀ALS_set{A1,A2,...,An} ∀petals{p1,p2,...,pn}:
                    [∀i ∈ {1,...,n}: almost_locked_set(Ai)]
                    ∧ [∀i ∈ {1,...,n}: pi ∈ candidates(cell_stem(rs,cs)) ∩ candidates(Ai)]
                    ∧ [∀i ∈ {1,...,n}: sees_ALS(cell_stem(rs,cs), Ai, pi)]
                    ∧ [∀i,j ∈ {1,...,n}, i≠j: pi ≠ pj]
                    → [∀cell(r,c): (∃i: sees_ALS(cell(r,c), Ai, pi)) 
                       ∧ sees(cell(r,c), cell_stem(rs,cs), pi) 
                       → remove_candidate(cell(r,c), pi)]
                """,
                "logic": "Stem cell connected to multiple ALS creates elimination opportunities",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["als_xz", "naked_triple", "hidden_triple"]
            },
            
            "multi_coloring": {
                "name": "Multi-Coloring",
                "description": "Extension of simple coloring to multiple color chains",
                "fol_rule": """
                ∀value(v) ∀color_chains{C1,C2,...,Cn} ∀colors{col1,col2,...,coln}:
                    [∀i ∈ {1,...,n}: conjugate_chain(Ci, v) ∧ two_coloring(Ci, coli)]
                    ∧ [∀i,j ∈ {1,...,n}, i≠j: Ci ∩ Cj = ∅]
                    ∧ [∃cell(r1,c1) ∈ Ci, cell(r2,c2) ∈ Cj: 
                       same_color_across_chains(coli, colj, cell(r1,c1), cell(r2,c2))
                       ∧ sees(cell(r1,c1), cell(r2,c2))]
                    → [eliminate_color_across_chains(coli, colj, common_color)]
                """,
                "logic": "Multiple color chains interaction for advanced eliminations",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["simple_coloring", "x_wing", "pointing_pairs"]
            },
            
            "sk_loop": {
                "name": "SK Loop",
                "description": "Strong link loop creating elimination pattern",
                "fol_rule": """
                ∀value(v) ∀cell_loop{(r1,c1),...,(rn,cn)} ∀link_types{l1,...,ln}:
                    [∀i ∈ {1,...,n}: strong_link(cell(ri,ci), cell(r(i+1),c(i+1)), v, li)]
                    ∧ [closed_loop((r1,c1),...,(rn,cn))]
                    ∧ [even_length(n)]
                    → [∀i ∈ {1,...,n}: weak_link(li) 
                       → ∀cell(r,c): sees(cell(r,c), cell(ri,ci)) ∧ sees(cell(r,c), cell(r(i+1),c(i+1)))
                       → remove_candidate(cell(r,c), v)]
                """,
                "logic": "Strong link loops create elimination opportunities at weak links",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["multi_coloring", "simple_coloring", "x_wing"]
            },
            
            "aic_discontinuous": {
                "name": "Alternating Inference Chain (Discontinuous)",
                "description": "Complex chain of strong and weak links with discontinuous eliminations",
                "fol_rule": """
                ∀chain{(r1,c1,v1),...,(rn,cn,vn)} ∀link_types{l1,...,l(n-1)}:
                    [∀i ∈ {1,...,n-1}: alternating_link(cell(ri,ci,vi), cell(r(i+1),c(i+1),v(i+1)), li)]
                    ∧ [alternating_pattern(l1,...,l(n-1))]
                    ∧ [first_link = strong ∧ last_link = strong]
                    → [∀cell(r,c): sees(cell(r,c), cell(r1,c1)) ∧ sees(cell(r,c), cell(rn,cn))
                       ∧ v1 = vn → remove_candidate(cell(r,c), v1)]
                """,
                "logic": "Alternating chains of strong/weak links for complex eliminations",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["sk_loop", "multi_coloring", "xy_wing"]
            },
            
            "als_chain": {
                "name": "ALS Chain",
                "description": "Chain of Almost Locked Sets connected by restricted common candidates",
                "fol_rule": """
                ∀ALS_chain{A1,A2,...,An} ∀connecting_values{v1,v2,...,v(n-1)}:
                    [∀i ∈ {1,...,n}: almost_locked_set(Ai)]
                    ∧ [∀i ∈ {1,...,n-1}: vi ∈ candidates(Ai) ∩ candidates(A(i+1))]
                    ∧ [∀i ∈ {1,...,n-1}: restricted_common(Ai, A(i+1), vi)]
                    ∧ [chain_forms_cycle(A1,...,An) ∨ chain_has_endpoints(A1,An)]
                    → [apply_als_elimination_rules(A1,...,An, v1,...,v(n-1))]
                """,
                "logic": "Chains of Almost Locked Sets for advanced pattern recognition",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["als_xz", "death_blossom", "hidden_triple"]
            },
            
            # "exocet": {
            #     "name": "Exocet",
            #     "description": "Advanced pattern with base cells and target cells in specific configuration",
            #     "fol_rule": """
            #     ∀base_cells{(rb1,cb1),(rb2,cb2)} ∀target_cells{(rt1,ct1),(rt2,ct2)} 
            #     ∀cross_lines{line1,line2} ∀values{v1,v2}:
            #         [candidates(base_cells) = {v1,v2}]
            #         ∧ [target_cells ∈ cross_lines] ∧ [base_cells ∉ cross_lines]
            #         ∧ [∀line ∈ cross_lines: ∀cell ∈ line ∖ target_cells: {v1,v2} ∩ candidates(cell) = ∅]
            #         ∧ [specific_box_constraints_satisfied(base_cells, target_cells)]
            #         → [apply_exocet_eliminations(base_cells, target_cells, v1, v2)]
            #     """,
            #     "logic": "Exocet pattern creates powerful eliminations through base-target relationships",
            #     "complexity": "hard",
            #     "composite": True,
            #     "composed_of": ["sue_de_coq", "als_chain", "box_line_reduction"]
            # },
            
            # "junior_exocet": {
            #     "name": "Junior Exocet",
            #     "description": "Simplified version of Exocet with relaxed constraints",
            #     "fol_rule": """
            #     ∀base_cells{(rb1,cb1),(rb2,cb2)} ∀target_cells{(rt1,ct1),(rt2,ct2)} 
            #     ∀cross_lines{line1,line2} ∀values{v1,v2}:
            #         [candidates(base_cells) ⊆ {v1,v2}] ∧ [|candidates(base_cells)| ≥ 2]
            #         ∧ [target_cells ∈ cross_lines] ∧ [base_cells ∉ cross_lines]
            #         ∧ [relaxed_cross_line_constraints(cross_lines, target_cells, v1, v2)]
            #         ∧ [junior_box_constraints_satisfied(base_cells, target_cells)]
            #         → [apply_junior_exocet_eliminations(base_cells, target_cells, v1, v2)]
            #     """,
            #     "logic": "Simplified Exocet with more accessible pattern recognition",
            #     "complexity": "hard",
            #     "composite": True,
            #     "composed_of": ["exocet", "sue_de_coq", "pointing_pairs"]
            # }
        }
    
    def get_strategy(self, strategy_name):
        return self.strategies.get(strategy_name)
    
    def list_strategies(self):
        return list(self.strategies.keys())
    
    def get_all_strategies(self):
        return self.strategies
    
    def get_easy_strategies(self):
        return self.easy_kb.get_all_strategies()
    
    def get_moderate_strategies(self):
        return self.moderate_kb.get_all_strategies()
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            return strategy.get('description', f'Strategy: {strategy_name}')
        return f'Strategy: {strategy_name}'

    def get_strategy_patterns(self, strategy_name: str) -> List[str]:
        """Get applicable patterns for this strategy"""
        # Return common patterns that this strategy can work with
        if 'row' in strategy_name:
            return ['row', 'horizontal']
        elif 'column' in strategy_name:
            return ['column', 'vertical']
        elif 'box' in strategy_name:
            return ['box', 'square']
        elif 'pair' in strategy_name:
            return ['pair', 'dual']
        elif 'wing' in strategy_name:
            return ['wing', 'triangle']
        else:
            return ['general', 'universal']

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy and strategy.get('composite', False):
            return strategy.get('composed_of', [])
        
        # For non-composite strategies, return basic prerequisites
        basic_prereqs = ['naked_single']
        if strategy_name not in basic_prereqs:
            return basic_prereqs
        return []