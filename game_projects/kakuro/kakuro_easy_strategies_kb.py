"""
Knowledge Base for Easy Kakuro Solving Strategies
Contains First-Order Logic (FOL) rules for basic Kakuro solving techniques
"""
from typing import List, Dict

class EasyStrategiesKB:
    def __init__(self):
        self.strategies = {
            "single_cell_sum": {
                "name": "Single Cell Sum",
                "description": "A cell has only one possible value based on its sum",
                "fol_rule": """
                ∀cell(r,c) ∀sum(s):
                    [sum(cell(r,c)) = s] ∧ [possible_values(s) = {v}] → [assign(cell(r,c), v)]
                """,
                "logic": "If a cell's sum can only be achieved with one value, assign that value",
                "complexity": "easy",
                "composite": False
            },
            
            "unique_sum_combination": {
                "name": "Unique Sum Combination",
                "description": "A sum can only be achieved with one combination of digits",
                "fol_rule": """
                ∀sum(s) ∀combination(c):
                    [possible_combinations(s) = {c}] → [assign_combination(c)]
                """,
                "logic": "If a sum can only be achieved with one combination of digits, use that combination",
                "complexity": "easy",
                "composite": False
            },
            
            "cross_reference": {
                "name": "Cross Reference",
                "description": "Use intersecting sums to narrow down possibilities",
                "fol_rule": """
                ∀cell(r,c) ∀sum_h(s_h) ∀sum_v(s_v):
                    [cell(r,c) ∈ sum_h] ∧ [cell(r,c) ∈ sum_v] 
                    → [possible_values(cell(r,c)) = intersection(possible_values(sum_h), possible_values(sum_v))]
                """,
                "logic": "Use the intersection of possible values from horizontal and vertical sums",
                "complexity": "easy",
                "composite": False
            },
            
            "eliminate_impossible": {
                "name": "Eliminate Impossible",
                "description": "Remove values that can't fit in a cell based on sum constraints",
                "fol_rule": """
                ∀cell(r,c) ∀value(v) ∀sum(s):
                    [cell(r,c) ∈ sum] ∧ [v ∉ possible_values_for_sum(s)] 
                    → [remove_candidate(cell(r,c), v)]
                """,
                "logic": "Remove values that can't be used to achieve the required sum",
                "complexity": "easy",
                "composite": False
            },
            
            "sum_partition": {
                "name": "Sum Partition",
                "description": "Break down a sum into possible partitions",
                "fol_rule": """
                ∀sum(s) ∀partition(p):
                    [valid_partition(s, p)] → [consider_partition(p)]
                """,
                "logic": "Consider all valid ways to partition a sum into digits",
                "complexity": "easy",
                "composite": False
            },
            
            "digit_frequency": {
                "name": "Digit Frequency",
                "description": "Track frequency of digits in possible combinations",
                "fol_rule": """
                ∀digit(d) ∀sum(s):
                    [frequency(d, s) = 1] → [digit_must_be_used(d, s)]
                """,
                "logic": "If a digit appears only once in possible combinations, it must be used",
                "complexity": "easy",
                "composite": False
            },
            
            "sum_difference": {
                "name": "Sum Difference",
                "description": "Use the difference between sums to find values",
                "fol_rule": """
                ∀sum1(s1) ∀sum2(s2) ∀cell(r,c):
                    [cell(r,c) ∈ s1] ∧ [cell(r,c) ∈ s2] 
                    → [possible_values(cell(r,c)) = values_in_range(s1 - s2)]
                """,
                "logic": "Use the difference between intersecting sums to find possible values",
                "complexity": "easy",
                "composite": False
            },
            
            "minimum_maximum": {
                "name": "Minimum/Maximum Values",
                "description": "Use minimum and maximum possible values for a sum",
                "fol_rule": """
                ∀sum(s) ∀cell(r,c):
                    [cell(r,c) ∈ s] → [value(cell(r,c)) ∈ [min_possible(s), max_possible(s)]]
                """,
                "logic": "Values must be within the minimum and maximum possible for their sum",
                "complexity": "easy",
                "composite": False
            },
            
            "sum_completion": {
                "name": "Sum Completion",
                "description": "Fill in the last cell of a sum when others are known",
                "fol_rule": """
                ∀sum(s) ∀cell(r,c):
                    [all_other_cells_filled(s)] ∧ [cell(r,c) ∈ s] 
                    → [assign(cell(r,c), remaining_value(s))]
                """,
                "logic": "If all but one cell in a sum are filled, fill the last cell",
                "complexity": "easy",
                "composite": False
            },
            
            "digit_elimination": {
                "name": "Digit Elimination",
                "description": "Remove digits that can't be used in a cell",
                "fol_rule": """
                ∀cell(r,c) ∀digit(d):
                    [d ∉ possible_digits_for_sum(sum(cell(r,c)))] 
                    → [remove_candidate(cell(r,c), d)]
                """,
                "logic": "Remove digits that can't be used to achieve the required sum",
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
        if 'sum' in strategy_name:
            return ['sum', 'total']
        elif 'digit' in strategy_name:
            return ['digit', 'number']
        elif 'cross' in strategy_name:
            return ['cross', 'intersection']
        elif 'partition' in strategy_name:
            return ['partition', 'combination']
        else:
            return ['general', 'universal']

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy and strategy.get('composite', False):
            return strategy.get('composed_of', [])
        
        # For non-composite strategies, return basic prerequisites
        basic_prereqs = ['single_cell_sum']
        if strategy_name not in basic_prereqs:
            return basic_prereqs
        return [] 