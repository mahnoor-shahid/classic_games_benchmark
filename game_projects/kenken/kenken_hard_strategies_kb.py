# kenken_hard_strategies_kb.py
"""
Knowledge Base for Hard Ken Ken Solving Strategies
Contains FOL rules that compose easy and moderate strategies for advanced techniques
"""

from kenken_easy_strategies_kb import KenKenEasyStrategiesKB
from kenken_moderate_strategies_kb import KenKenModerateStrategiesKB
from typing import List, Dict

class KenKenHardStrategiesKB:
    def __init__(self):
        self.easy_kb = KenKenEasyStrategiesKB()
        self.moderate_kb = KenKenModerateStrategiesKB()
        self.strategies = {
            "advanced_cage_chaining": {
                "name": "Advanced Cage Chaining",
                "description": "Chain constraints across multiple interconnected cages",
                "fol_rule": """
                ∀cage_chain{c1,c2,...,cn} ∀connecting_cells{s1,s2,...,s(n-1)} ∀constraints{con1,...,conn}:
                    [∀i ∈ {1,...,n-1}: cells_in_cage(ci) ∩ cells_in_cage(c(i+1)) = {si}]
                    ∧ [∀i: cage_constraint(ci) = coni]
                    → [propagate_chain_constraints(cage_chain, connecting_cells, constraints)]
                """,
                "logic": "Propagate constraints through chains of connected cages",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["multi_cage_analysis", "constraint_propagation", "cage_intersection"]
            },
            
            "complex_factorization_patterns": {
                "name": "Complex Factorization Patterns",
                "description": "Use advanced number theory for complex multiplication/division patterns",
                "fol_rule": """
                ∀cage_set{c1,c2,...,ck} ∀target_set{t1,t2,...,tk} ∀operations{op1,op2,...,opk}:
                    [∀i: cage_operation(ci) ∈ {multiply,divide}] ∧ [∀i: cage_target(ci) = ti]
                    ∧ [mathematical_relationship_exists(t1,t2,...,tk)]
                    → [exploit_number_theoretic_properties(cage_set, target_set, operations)]
                """,
                "logic": "Exploit number theoretic relationships between multiplication/division targets",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["advanced_factorization", "quotient_remainder_analysis", "complex_arithmetic_cages"]
            },
            
            "contradiction_analysis": {
                "name": "Contradiction Analysis",
                "description": "Find solutions by systematically exploring and eliminating contradictions",
                "fol_rule": """
                ∀assumption(A) ∀grid_state(G) ∀contradiction(CONTR):
                    [make_assumption(A, G)] ∧ [propagate_fully(G, A)]
                    ∧ [leads_to_contradiction(G, A, CONTR)]
                    → [eliminate_assumption(A) ∧ assert_negation(¬A)]
                """,
                "logic": "Use proof by contradiction to eliminate impossible assignments",
                "complexity": "hard",
                "composite": True,
                "composed_of": ["elimination_by_exhaustion", "constraint_propagation", "global_constraint_optimization"]
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
        if 'chaining' in strategy_name or 'chain' in strategy_name:
            return ['chaining', 'cascade', 'propagation']
        elif 'factorization' in strategy_name:
            return ['factorization', 'number_theory', 'mathematical']
        elif 'constraint' in strategy_name and 'satisfaction' in strategy_name:
            return ['csp', 'constraint_satisfaction', 'algorithm']
        elif 'decomposition' in strategy_name:
            return ['decomposition', 'recursive', 'divide_conquer']
        elif 'optimization' in strategy_name:
            return ['optimization', 'global', 'constraint']
        elif 'pattern' in strategy_name:
            return ['pattern', 'recognition', 'template']
        elif 'induction' in strategy_name:
            return ['induction', 'mathematical', 'proof']
        elif 'contradiction' in strategy_name:
            return ['contradiction', 'proof', 'elimination']
        elif 'symmetry' in strategy_name:
            return ['symmetry', 'group_theory', 'transformation']
        elif 'meta' in strategy_name:
            return ['meta', 'reasoning', 'higher_order']
        elif 'temporal' in strategy_name:
            return ['temporal', 'ordering', 'sequence']
        elif 'probabilistic' in strategy_name:
            return ['probabilistic', 'bayesian', 'statistical']
        else:
            return ['hard', 'advanced', 'complex']

    def get_strategy_prerequisites(self, strategy_name: str) -> List[str]:
        """Get prerequisite strategies for this strategy"""
        strategy = self.strategies.get(strategy_name)
        if strategy and strategy.get('composite', False):
            return strategy.get('composed_of', [])
        
        # For non-composite strategies, return basic prerequisites
        basic_prereqs = ['naked_single', 'single_cell_cage', 'constraint_propagation']
        if strategy_name not in basic_prereqs:
            return basic_prereqs
        return []
    
    def get_operations_used(self, strategy_name: str) -> List[str]:
        """Get arithmetic operations used by this strategy"""
        if 'factorization' in strategy_name:
            return ['multiply', 'divide']
        elif 'arithmetic' in strategy_name:
            return ['add', 'subtract', 'multiply', 'divide']
        elif any(term in strategy_name for term in ['sum', 'addition']):
            return ['add']
        elif any(term in strategy_name for term in ['product', 'multiplication']):
            return ['multiply']
        elif any(term in strategy_name for term in ['quotient', 'division']):
            return ['divide']
        else:
            return []  # Meta-strategies may not use specific operations