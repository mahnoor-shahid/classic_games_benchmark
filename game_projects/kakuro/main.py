"""
Kakuro Puzzle Generator and Solver
Main entry point for the Kakuro game
"""

import argparse
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from template_based_generators import TemplateBasedGenerator
from kakuro_validator import KakuroValidator
from kakuro_easy_strategies_kb import EasyStrategiesKB
from kakuro_moderate_strategies_kb import ModerateStrategiesKB
from kakuro_hard_strategies_kb import HardStrategiesKB

class KakuroGame:
    def __init__(self):
        self.generator = TemplateBasedGenerator()
        self.validator = KakuroValidator()
        self.easy_kb = EasyStrategiesKB()
        self.moderate_kb = ModerateStrategiesKB()
        self.hard_kb = HardStrategiesKB()
        
        # Create output directories
        os.makedirs('output/kakuro/puzzles', exist_ok=True)
        os.makedirs('output/kakuro/solutions', exist_ok=True)
        os.makedirs('output/kakuro/analysis', exist_ok=True)
        
        print("✅ Kakuro game initialized")

    def generate_validated_puzzles(self, difficulty: str, count: int) -> List[Dict]:
        """Generate and validate puzzles"""
        print(f"\n🎲 Generating {count} {difficulty} Kakuro puzzles...")
        
        puzzles = self.generator.generate_puzzles(difficulty, count)
        validated_puzzles = []
        
        for i, puzzle in enumerate(puzzles, 1):
            print(f"\n🔍 Validating puzzle {i}/{len(puzzles)}...")
            
            # Generate solution
            solution = self._solve_puzzle(puzzle)
            
            if solution is not None:
                # Validate puzzle
                if self.validator.validate_puzzle(puzzle, solution):
                    validated_puzzles.append({
                        'puzzle': puzzle,
                        'solution': solution.tolist()
                    })
                    print(f"✅ Puzzle {i} validated successfully")
                else:
                    print(f"❌ Puzzle {i} failed validation")
            else:
                print(f"❌ Failed to solve puzzle {i}")
        
        return validated_puzzles

    def _solve_puzzle(self, puzzle: Dict) -> Optional[np.ndarray]:
        """Solve a Kakuro puzzle"""
        # TODO: Implement puzzle solving logic
        # For now, return a dummy solution for testing
        grid = np.array(puzzle['grid'])
        solution = np.zeros_like(grid)
        
        # Fill in some values for testing
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == -1:
                    solution[i, j] = 1  # Dummy value
        
        return solution

    def save_puzzles(self, puzzles: List[Dict], difficulty: str):
        """Save puzzles and solutions to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save puzzles
        puzzles_file = f'output/kakuro/puzzles/kakuro_{difficulty}_{timestamp}.json'
        with open(puzzles_file, 'w') as f:
            json.dump(puzzles, f, indent=2)
        
        print(f"\n💾 Saved {len(puzzles)} puzzles to {puzzles_file}")
        
        # Generate and save analysis
        self._generate_analysis(puzzles, difficulty, timestamp)

    def _generate_analysis(self, puzzles: List[Dict], difficulty: str, timestamp: str):
        """Generate analysis of the puzzles"""
        analysis = {
            'total_puzzles': len(puzzles),
            'difficulty': difficulty,
            'generation_time': self.generator.get_stats()['total_time'],
            'validation_stats': self.validator.get_validation_stats(),
            'templates_used': {},
            'strategy_usage': {}
        }
        
        # Analyze template usage
        for puzzle in puzzles:
            template = puzzle['puzzle']['template']
            analysis['templates_used'][template] = analysis['templates_used'].get(template, 0) + 1
        
        # Analyze strategy usage
        for puzzle in puzzles:
            for strategy in puzzle['puzzle']['required_strategies']:
                analysis['strategy_usage'][strategy] = analysis['strategy_usage'].get(strategy, 0) + 1
        
        # Save analysis
        analysis_file = f'output/kakuro/analysis/kakuro_{difficulty}_{timestamp}_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📊 Analysis saved to {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description='Kakuro Puzzle Generator')
    parser.add_argument('--action', type=str, required=True,
                      choices=['generate_validated'],
                      help='Action to perform')
    parser.add_argument('--difficulty', type=str, default='easy',
                      choices=['easy', 'moderate', 'hard'],
                      help='Difficulty level')
    parser.add_argument('--count', type=int, default=10,
                      help='Number of puzzles to generate')
    
    args = parser.parse_args()
    
    game = KakuroGame()
    
    if args.action == 'generate_validated':
        puzzles = game.generate_validated_puzzles(args.difficulty, args.count)
        game.save_puzzles(puzzles, args.difficulty)
    else:
        print(f"❌ Unknown action: {args.action}")

if __name__ == '__main__':
    main() 