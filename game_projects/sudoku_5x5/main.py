import os
import sys
import argparse
import json
import traceback
import numpy as np
from template_based_generators import TemplateBasedGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_puzzles_as_json(puzzles, fname):
    with open(fname, "w") as f:
        json.dump([puzzle[0].tolist() for puzzle in puzzles], f)

def wrap_puzzles_for_metadata(puzzles, prefix):
    return [{
        'puzzle_grid': puzzle.tolist(),
        'solution_grid': solution.tolist(),
        'difficulty': 'unvalidated',
        'id': f"{prefix}_{i:04d}"
    } for i, (puzzle, solution) in enumerate(puzzles)]

def load_config_num_puzzles(difficulty):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return int(config['generation']['num_puzzles'].get(difficulty, 5))
    except Exception:
        return 5

def main():
    parser = argparse.ArgumentParser(description="MNIST 5x5 Sudoku Project")
    parser.add_argument('--action', choices=[
        'generate_validated', 'generate_unvalidated'
    ], required=True, help='Action to perform')
    parser.add_argument('--difficulty', choices=['easy', 'moderate', 'hard'], default='easy')
    parser.add_argument('--num_clues', type=int, default=12, help='Number of clues for unvalidated puzzles')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    images_dir = os.path.join(base_dir, "images")
    ensure_dir(output_dir)
    ensure_dir(images_dir)

    template_generator = TemplateBasedGenerator()

    if args.action == "generate_unvalidated":
        num_puzzles = load_config_num_puzzles(args.difficulty)
        print(f"\n⚡ Generating {num_puzzles} unvalidated 5x5 Sudoku puzzles for 0-4 and 5-9 digits...")
        results = template_generator.generate_unvalidated_puzzles(num_0_4=num_puzzles, num_5_9=num_puzzles, num_clues=args.num_clues)
        save_puzzles_as_json(results['0_4'], os.path.join(output_dir, f"sudoku5x5_0_4_unvalidated.json"))
        save_puzzles_as_json(results['5_9'], os.path.join(output_dir, f"sudoku5x5_5_9_unvalidated.json"))
        print(f"Saved {len(results['0_4'])} puzzles for digits 0-4 and {len(results['5_9'])} for 5-9.")
        if results['0_4']:
            puzzles_0_4 = wrap_puzzles_for_metadata(results['0_4'], "unvalidated_0_4")
            template_generator.save_mnist_images_with_metadata(puzzles_0_4, os.path.join(images_dir, "0_4_unvalidated"))
        if results['5_9']:
            puzzles_5_9 = wrap_puzzles_for_metadata(results['5_9'], "unvalidated_5_9")
            template_generator.save_mnist_images_with_metadata(puzzles_5_9, os.path.join(images_dir, "5_9_unvalidated"))

    elif args.action == "generate_validated":
        print("\n🔬 GENERATING & VALIDATING 5x5 SUDOKU PUZZLES")
        print("Validated puzzle generation not implemented in this minimal rewrite.")
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)