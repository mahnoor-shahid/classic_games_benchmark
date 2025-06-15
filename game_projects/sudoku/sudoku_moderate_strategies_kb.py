# moderate_strategies_kb.py
"""
Knowledge Base for Moderate Sudoku Solving Strategies
Contains FOL rules that compose easy strategies for intermediate techniques
"""

from sudoku_easy_strategies_kb import EasyStrategiesKB
from typing import List

class ModerateStrategiesKB:
    def __init__(self):
        self.easy_kb = EasyStrategiesKB()
        self.strategies = {
            "naked_pair": {
                "name": "Naked Pair",
                "description": "Two cells in the same unit have the same two candidates",
                "fol_rule": """
                ∀unit(u) ∀cell(r1,c1) ∀cell(r2,c2) ∀value_set{v1,v2}:
                    [cell(r1,c1) ∈ unit(u)] ∧ [cell(r2,c2) ∈ unit(u)] ∧ [(r1,c1) ≠ (r2,c2)]
                    ∧ [candidates(cell(r1,c1)) = {v1,v2}] ∧ [candidates(cell(r2,c2)) = {v1,v2}]
                    → [∀cell(r',c') ∈ unit(u), (r',c') ∉ {(r1,c1),(r2,c2)}: 
                       remove_candidate(cell(r',c'), v1) ∧ remove_candidate(cell(r',c'), v2)]
                """,
                "logic": "If two cells in a unit have identical two-candidate sets, remove those candidates from other cells in the unit",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["eliminate_candidates_row", "eliminate_candidates_column", "eliminate_candidates_box"]
            },
            
            "naked_triple": {
                "name": "Naked Triple",
                "description": "Three cells in the same unit collectively contain only three candidates",
                "fol_rule": """
                ∀unit(u) ∀cell_set{(r1,c1),(r2,c2),(r3,c3)} ∀value_set{v1,v2,v3}:
                    [∀i ∈ {1,2,3}: cell(ri,ci) ∈ unit(u)] ∧ [all_distinct((r1,c1),(r2,c2),(r3,c3))]
                    ∧ [⋃{candidates(cell(ri,ci)) : i ∈ {1,2,3}} = {v1,v2,v3}]
                    → [∀cell(r',c') ∈ unit(u), (r',c') ∉ {(r1,c1),(r2,c2),(r3,c3)}: 
                       ∀v ∈ {v1,v2,v3}: remove_candidate(cell(r',c'), v)]
                """,
                "logic": "If three cells in a unit collectively contain only three candidates, remove those candidates from other cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["naked_pair", "eliminate_candidates_row", "eliminate_candidates_column", "eliminate_candidates_box"]
            },
            
            "hidden_pair": {
                "name": "Hidden Pair",
                "description": "Two values that can only go in two specific cells within a unit",
                "fol_rule": """
                ∀unit(u) ∀value_set{v1,v2} ∀cell_set{(r1,c1),(r2,c2)}:
                    [∀v ∈ {v1,v2}: ∀cell(r,c) ∈ unit(u): 
                     v ∈ candidates(cell(r,c)) → (r,c) ∈ {(r1,c1),(r2,c2)}]
                    → [∀i ∈ {1,2}: candidates(cell(ri,ci)) := candidates(cell(ri,ci)) ∩ {v1,v2}]
                """,
                "logic": "If two values can only be placed in two cells, remove all other candidates from those cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_single_row", "hidden_single_column", "hidden_single_box"]
            },
            
            "hidden_triple": {
                "name": "Hidden Triple",
                "description": "Three values that can only go in three specific cells within a unit",
                "fol_rule": """
                ∀unit(u) ∀value_set{v1,v2,v3} ∀cell_set{(r1,c1),(r2,c2),(r3,c3)}:
                    [∀v ∈ {v1,v2,v3}: ∀cell(r,c) ∈ unit(u): 
                     v ∈ candidates(cell(r,c)) → (r,c) ∈ {(r1,c1),(r2,c2),(r3,c3)}]
                    → [∀i ∈ {1,2,3}: candidates(cell(ri,ci)) := candidates(cell(ri,ci)) ∩ {v1,v2,v3}]
                """,
                "logic": "If three values can only be placed in three cells, remove all other candidates from those cells",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_pair", "hidden_single_row", "hidden_single_column", "hidden_single_box"]
            },
            
            "pointing_pairs": {
                "name": "Pointing Pairs/Triples",
                "description": "Candidates in a box that align in a row/column eliminate candidates in that row/column",
                "fol_rule": """
                ∀box(b) ∀row(r) ∀value(v):
                    [∀cell(r',c) ∈ box(b): v ∈ candidates(cell(r',c)) → r' = r]
                    → [∀cell(r,c') ∉ box(b): remove_candidate(cell(r,c'), v)]
                ∨
                ∀box(b) ∀column(c) ∀value(v):
                    [∀cell(r,c') ∈ box(b): v ∈ candidates(cell(r,c')) → c' = c]
                    → [∀cell(r',c) ∉ box(b): remove_candidate(cell(r',c), v)]
                """,
                "logic": "If all candidates for a value in a box lie on the same row/column, eliminate that value from the rest of the row/column",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["eliminate_candidates_row", "eliminate_candidates_column", "eliminate_candidates_box"]
            },
            
            "box_line_reduction": {
                "name": "Box/Line Reduction",
                "description": "If a value in a row/column can only be in one box, eliminate it from other cells in that box",
                "fol_rule": """
                ∀row(r) ∀box(b) ∀value(v):
                    [∀cell(r,c): v ∈ candidates(cell(r,c)) → cell(r,c) ∈ box(b)]
                    → [∀cell(r',c') ∈ box(b), r' ≠ r: remove_candidate(cell(r',c'), v)]
                ∨
                ∀column(c) ∀box(b) ∀value(v):
                    [∀cell(r,c): v ∈ candidates(cell(r,c)) → cell(r,c) ∈ box(b)]
                    → [∀cell(r',c') ∈ box(b), c' ≠ c: remove_candidate(cell(r',c'), v)]
                """,
                "logic": "If candidates for a value in a line are confined to one box, eliminate from other cells in that box",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["pointing_pairs", "eliminate_candidates_box"]
            },
            
            "xy_wing": {
                "name": "XY-Wing",
                "description": "Three cells forming a Y-shape with specific candidate patterns",
                "fol_rule": """
                ∀cell_pivot(rp,cp) ∀cell_wing1(r1,c1) ∀cell_wing2(r2,c2) ∀values{x,y,z}:
                    [candidates(cell_pivot(rp,cp)) = {x,y}] 
                    ∧ [candidates(cell_wing1(r1,c1)) = {x,z}] 
                    ∧ [candidates(cell_wing2(r2,c2)) = {y,z}]
                    ∧ [sees(cell_pivot(rp,cp), cell_wing1(r1,c1))] 
                    ∧ [sees(cell_pivot(rp,cp), cell_wing2(r2,c2))]
                    ∧ [¬sees(cell_wing1(r1,c1), cell_wing2(r2,c2))]
                    → [∀cell(r,c): sees(cell_wing1(r1,c1), cell(r,c)) ∧ sees(cell_wing2(r2,c2), cell(r,c)) 
                       → remove_candidate(cell(r,c), z)]
                """,
                "logic": "In XY-Wing pattern, eliminate the common wing value from cells that see both wings",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["naked_pair", "eliminate_candidates_row", "eliminate_candidates_column", "eliminate_candidates_box"]
            },
            
            "simple_coloring": {
                "name": "Simple Coloring",
                "description": "Color conjugate pairs to find contradictions",
                "fol_rule": """
                ∀value(v) ∀cell_set(C) ∀color_function(f):
                    [∀cell(r,c) ∈ C: |candidates_for_value(v, unit_containing(cell(r,c)))| = 2]
                    ∧ [conjugate_chain(C, v)] ∧ [two_coloring(C, f)]
                    → [∀cell(r1,c1), cell(r2,c2): same_color(f, cell(r1,c1), cell(r2,c2)) 
                       ∧ sees(cell(r1,c1), cell(r2,c2)) → eliminate_color(f, color_of(cell(r1,c1)))]
                """,
                "logic": "Use conjugate pairs to create color chains; eliminate colors that create contradictions",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["hidden_single_row", "hidden_single_column", "hidden_single_box"]
            },
            
            "x_wing": {
                "name": "X-Wing",
                "description": "Four cells forming a rectangle with a value appearing only at corners in two rows/columns",
                "fol_rule": """
                ∀rows{r1,r2} ∀columns{c1,c2} ∀value(v):
                    [r1 ≠ r2] ∧ [c1 ≠ c2]
                    ∧ [∀c ∈ {c1,c2}: ∀r ∈ {r1,r2}: v ∈ candidates(cell(r,c))]
                    ∧ [∀c ∉ {c1,c2}: ∀r ∈ {r1,r2}: v ∉ candidates(cell(r,c))]
                    → [∀r ∉ {r1,r2}: ∀c ∈ {c1,c2}: remove_candidate(cell(r,c), v)]
                ∨
                ∀columns{c1,c2} ∀rows{r1,r2} ∀value(v):
                    [c1 ≠ c2] ∧ [r1 ≠ r2]
                    ∧ [∀r ∈ {r1,r2}: ∀c ∈ {c1,c2}: v ∈ candidates(cell(r,c))]
                    ∧ [∀r ∉ {r1,r2}: ∀c ∈ {c1,c2}: v ∉ candidates(cell(r,c))]
                    → [∀c ∉ {c1,c2}: ∀r ∈ {r1,r2}: remove_candidate(cell(r,c), v)]
                """,
                "logic": "If a value forms a rectangle pattern in two rows/columns, eliminate it from intersecting lines",
                "complexity": "moderate",
                "composite": True,
                "composed_of": ["pointing_pairs", "box_line_reduction"]
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