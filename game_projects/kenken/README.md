# KenKen Puzzle Generator

A sophisticated KenKen puzzle generator that creates arithmetic puzzles with varying difficulty levels and operation types.

## 🌟 Features

- **Arithmetic Cage Generation**: Creates valid arithmetic cages with multiple operations
- **Multiple Operation Types**: Supports addition, subtraction, multiplication, and division
- **Difficulty Levels**: Easy, moderate, and hard puzzles with appropriate complexity
- **Strategy Validation**: Ensures puzzles can be solved using specific strategies
- **Quality Assurance**: Comprehensive validation of puzzle quality and solvability
- **Configurable Generation**: Extensive configuration options for fine-tuning puzzle generation

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages (see Installation section)

## 🚀 Installation

1. Navigate to the KenKen directory:
```bash
cd game_projects/kenken
```

2. Create and activate a virtual environment:
```bash
python -m venv kenken-env
source kenken-env/bin/activate  # On Windows: kenken-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage

Generate KenKen puzzles:
```bash
python kenken_main.py --action generate
```

### Available Actions

- `generate`: Generate new KenKen puzzles
- `analyze`: Analyze existing puzzle datasets
- `test_solver`: Test the solver on sample puzzles
- `create_sample`: Create a sample puzzle of specified difficulty

### Configuration

The project uses `kenken_config.yaml` for configuration. Key settings include:

- **Generation Settings**:
  - Grid size
  - Operation types
  - Cage sizes
  - Difficulty parameters

- **Validation Settings**:
  - Quality thresholds
  - Solvability checks
  - Strategy validation
  - Uniqueness verification

## 🏗️ Project Structure

```
kenken/
├── kenken_main.py           # Main orchestration script
├── kenken_config.yaml       # Configuration file
├── kenken_generator.py      # Core generation logic
├── kenken_solver.py         # Puzzle solving implementation
├── kenken_analyzer.py       # Dataset analysis tools
├── kenken_config_manager.py # Configuration management
├── kenken_template_generators.py  # Template-based generation
└── strategy_kb/             # Strategy knowledge bases
    ├── kenken_easy_strategies_kb.py
    ├── kenken_moderate_strategies_kb.py
    └── kenken_hard_strategies_kb.py
```

## 🧩 Puzzle Generation Process

1. **Grid Generation**:
   - Creates base grid
   - Determines cage boundaries
   - Assigns operations

2. **Cage Creation**:
   - Generates valid arithmetic cages
   - Ensures unique solutions
   - Maintains difficulty level

3. **Strategy Integration**:
   - Ensures required strategies can be applied
   - Validates strategy sequence
   - Checks compositionality

4. **Quality Validation**:
   - Verifies puzzle solvability
   - Checks difficulty level
   - Ensures human-solvable complexity

## 🎯 Difficulty Levels

### Easy Puzzles
- 4x4 or 5x5 grid
- Simple operations (mostly addition)
- Small cage sizes
- Clear patterns

### Moderate Puzzles
- 5x5 or 6x6 grid
- Mixed operations
- Medium cage sizes
- Balanced complexity

### Hard Puzzles
- 6x6 or larger grid
- Complex operations
- Large cage sizes
- Minimal clues

## 🔍 Validation Process

1. **Basic Validation**:
   - Grid structure
   - Cage validity
   - Operation correctness

2. **Strategy Validation**:
   - Required strategy presence
   - Strategy sequence
   - Compositionality

3. **Quality Checks**:
   - Solvability
   - Uniqueness
   - Difficulty assessment

## 📊 Output

- **Puzzle Files**:
  - JSON format
  - Includes grid, cages, and operations
  - Strategy information

- **Images**:
  - Puzzle grids
  - Solution grids
  - Analysis visualizations

- **Analysis Reports**:
  - Strategy usage
  - Difficulty distribution
  - Quality metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- KenKen puzzle community
- Open source contributors

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker. 