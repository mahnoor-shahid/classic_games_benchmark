# Futoshiki Puzzle Generator

A sophisticated Futoshiki puzzle generator that creates inequality-based puzzles with varying sizes and difficulty levels.

## 🌟 Features

- **Inequality Constraint Generation**: Creates valid inequality constraints between cells
- **Variable Grid Sizes**: Supports puzzles from 4x4 to 9x9
- **Difficulty Levels**: Easy, moderate, and hard puzzles with appropriate complexity
- **Strategy Validation**: Ensures puzzles can be solved using specific strategies
- **Quality Assurance**: Comprehensive validation of puzzle quality and solvability
- **Configurable Generation**: Extensive configuration options for fine-tuning puzzle generation

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages (see Installation section)

## 🚀 Installation

1. Navigate to the Futoshiki directory:
```bash
cd game_projects/futoshiki
```

2. Create and activate a virtual environment:
```bash
python -m venv futoshiki-env
source futoshiki-env/bin/activate  # On Windows: futoshiki-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage

Generate Futoshiki puzzles:
```bash
python futoshiki_main.py --action generate
```

### Available Actions

- `generate`: Generate new Futoshiki puzzles
- `analyze`: Analyze existing puzzle datasets
- `test_solver`: Test the solver on sample puzzles
- `create_sample`: Create a sample puzzle of specified difficulty

### Configuration

The project uses `futoshiki_config.yaml` for configuration. Key settings include:

- **Generation Settings**:
  - Grid size
  - Constraint density
  - Initial numbers
  - Difficulty parameters

- **Validation Settings**:
  - Quality thresholds
  - Solvability checks
  - Strategy validation
  - Uniqueness verification

## 🏗️ Project Structure

```
futoshiki/
├── futoshiki_main.py           # Main orchestration script
├── futoshiki_config.yaml       # Configuration file
├── futoshiki_generator.py      # Core generation logic
├── futoshiki_solver.py         # Puzzle solving implementation
├── futoshiki_analyzer.py       # Dataset analysis tools
├── futoshiki_config_manager.py # Configuration management
├── futoshiki_template_generator.py  # Template-based generation
├── futoshiki_template_generator_enhanced.py  # Enhanced generation
├── enhanced_image_generator.py # Image generation utilities
└── strategy_kb/                # Strategy knowledge bases
    ├── futoshiki_easy_strategies_kb.py
    ├── futoshiki_moderate_strategies_kb.py
    └── futoshiki_hard_strategies_kb.py
```

## 🧩 Puzzle Generation Process

1. **Grid Generation**:
   - Creates base grid
   - Determines initial numbers
   - Places inequality constraints

2. **Constraint Creation**:
   - Generates valid inequality constraints
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
- Simple constraints
- More initial numbers
- Clear patterns

### Moderate Puzzles
- 5x5 or 6x6 grid
- Mixed constraints
- Balanced initial numbers
- Moderate complexity

### Hard Puzzles
- 6x6 or larger grid
- Complex constraints
- Fewer initial numbers
- Minimal clues

## 🔍 Validation Process

1. **Basic Validation**:
   - Grid structure
   - Constraint validity
   - Number placement

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
  - Includes grid and constraints
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

- Futoshiki puzzle community
- Open source contributors

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker. 