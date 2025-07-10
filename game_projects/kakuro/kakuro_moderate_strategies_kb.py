"""
Kakuro Moderate Strategies Knowledge Base
Contains strategies for solving moderate difficulty Kakuro puzzles
"""

from typing import Dict, List, Optional

class ModerateStrategiesKB:
    def __init__(self):
        self.strategies = {
            'sum_partition': {
                'name': 'Sum Partition',
                'description': 'Break down a sum into possible partitions of digits',
                'first_order_logic': '∀s ∈ Sums, ∃p ∈ Partitions(s) : ValidPartition(p)',
                'complexity': 'moderate',
                'prerequisites': ['single_cell_sum', 'unique_sum_combination']
            },
            'digit_frequency': {
                'name': 'Digit Frequency',
                'description': 'Track frequency of digits in combinations to identify patterns',
                'first_order_logic': '∀d ∈ Digits, CountFrequency(d) → IdentifyPattern(d)',
                'complexity': 'moderate',
                'prerequisites': ['sum_partition']
            },
            'sum_difference': {
                'name': 'Sum Difference',
                'description': 'Use difference between sums to find values',
                'first_order_logic': '∀s1,s2 ∈ Sums, Difference(s1,s2) → InferValue(s1,s2)',
                'complexity': 'moderate',
                'prerequisites': ['sum_partition', 'digit_frequency']
            },
            'minimum_maximum': {
                'name': 'Minimum Maximum',
                'description': 'Ensure values are within min/max for their sum',
                'first_order_logic': '∀s ∈ Sums, MinMax(s) → ConstrainValues(s)',
                'complexity': 'moderate',
                'prerequisites': ['sum_difference']
            },
            'sum_completion': {
                'name': 'Sum Completion',
                'description': 'Fill in last cell of a sum when others are known',
                'first_order_logic': '∀s ∈ Sums, KnownCells(s) → CompleteSum(s)',
                'complexity': 'moderate',
                'prerequisites': ['minimum_maximum']
            },
            'digit_elimination': {
                'name': 'Digit Elimination',
                'description': 'Remove digits that cannot be used in a cell',
                'first_order_logic': '∀c ∈ Cells, ImpossibleDigits(c) → EliminateDigits(c)',
                'complexity': 'moderate',
                'prerequisites': ['sum_completion']
            }
        }

    def get_strategy(self, strategy_name: str) -> Optional[Dict]:
        """Get a strategy by name"""
        return self.strategies.get(strategy_name)

    def list_strategies(self) -> List[str]:
        """List all available strategies"""
        return list(self.strategies.keys())

    def get_strategy_description(self, strategy_name: str) -> Optional[str]:
        """Get the description of a strategy"""
        strategy = self.get_strategy(strategy_name)
        return strategy['description'] if strategy else None

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get the prerequisites for a strategy"""
        strategy = self.get_strategy(strategy_name)
        return strategy['prerequisites'] if strategy else []

    def get_strategy_complexity(self, strategy_name: str) -> Optional[str]:
        """Get the complexity level of a strategy"""
        strategy = self.get_strategy(strategy_name)
        return strategy['complexity'] if strategy else None

    def get_strategy_logic(self, strategy_name: str) -> Optional[str]:
        """Get the first-order logic representation of a strategy"""
        strategy = self.get_strategy(strategy_name)
        return strategy['first_order_logic'] if strategy else None

    def validate_strategy_sequence(self, sequence: List[str]) -> bool:
        """Validate if a sequence of strategies is valid based on prerequisites"""
        if not sequence:
            return False

        # Check if all strategies exist
        if not all(s in self.strategies for s in sequence):
            return False

        # Check prerequisites for each strategy
        for i, strategy in enumerate(sequence):
            prerequisites = self.get_strategy_prerequisites(strategy)
            if not all(p in sequence[:i] for p in prerequisites):
                return False

        return True

    def get_available_strategies(self, completed_strategies: List[str]) -> List[str]:
        """Get strategies that can be used next based on completed strategies"""
        available = []
        for strategy in self.strategies:
            prerequisites = self.get_strategy_prerequisites(strategy)
            if all(p in completed_strategies for p in prerequisites):
                available.append(strategy)
        return available 