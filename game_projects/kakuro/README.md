# Kakuro Puzzle Generator

A template-based Kakuro puzzle generator that creates puzzles with varying difficulty levels, incorporating MNIST digit recognition and strategy-based solving approaches.

## Overview

This module generates Kakuro puzzles with the following features:
- Template-based generation for consistent puzzle patterns
- Multiple difficulty levels (easy, moderate, hard)
- MNIST digit integration for realistic number representation
- Strategy-based solving approaches
- Compositionality validation
- Symmetry patterns
- Analysis and statistics generation

## Structure

```
kakuro/
├── main.py                 # Main entry point
├── config.yaml            # Configuration settings
├── kakuro_validator.py    # Puzzle validation logic
├── template_based_generators.py  # Puzzle generation
├── kakuro_easy_strategies_kb.py    # Easy strategies knowledge base
├── kakuro_moderate_strategies_kb.py # Moderate strategies knowledge base
└── kakuro_hard_strategies_kb.py    # Hard strategies knowledge base
```

## Features

### Puzzle Generation
- Template-based generation for consistent patterns
- Multiple symmetry types (cross, border, checkerboard, diamond, spiral, complex)
- MNIST digit integration for realistic number representation
- Configurable grid sizes and difficulty levels

### Strategy System
- Easy strategies:
  - Single cell sum
  - Unique sum combination
  - Cross reference
  - Eliminate impossible
  - Sum partition
  - Digit frequency

- Moderate strategies:
  - Sum partition
  - Digit frequency
  - Sum difference
  - Minimum maximum
  - Sum completion
  - Digit elimination

- Hard strategies:
  - Sum completion
  - Digit elimination
  - Sum difference
  - Minimum maximum
  - Cross reference
  - Unique sum combination

### Validation
- Grid structure validation
- Sum validation (horizontal and vertical)
- Compositionality validation
- Symmetry validation
- Strategy sequence validation
- Cell relationship validation

### Output
- JSON format for puzzles and solutions
- Analysis reports with statistics
- MNIST-based digit images
- Template usage analysis
- Strategy usage analysis

## Usage

```bash
# Generate validated puzzles
python main.py --action generate_validated 
```

## Configuration

The `config.yaml` file contains settings for:
- Grid dimensions
- Difficulty levels
- Generation parameters
- Validation rules
- Output settings
- MNIST integration

## Dependencies

- NumPy
- OpenCV
- scikit-learn
- tqdm
- PyYAML

## Output Format

### Puzzle JSON
```json
{
  "grid": [[...]],  # 2D array of numbers and sums
  "template": "cross",  # Template type used
  "difficulty": "easy",
  "required_strategies": ["single_cell_sum", "unique_sum_combination"],
  "solution": [[...]]  # 2D array of solution numbers
}
```

### Analysis JSON
```json
{
  "total_puzzles": 10,
  "difficulty": "easy",
  "generation_time": 120.5,
  "validation_stats": {...},
  "templates_used": {...},
  "strategy_usage": {...}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 