# MNIST 5x5 Sudoku Generator

A variant of the original MNIST Sudoku generator, this project generates 5x5 Sudoku puzzles using only the digits 0-4 (from MNIST). It is based on the original 9x9 project, with changes to support the 5x5 grid and digit set.

## Differences from the 9x9 Project

- **Grid Size:** 5x5 instead of 9x9
- **Digits Used:** 0, 1, 2, 3, 4 (instead of 1-9)
- **Validation and Generation:** All logic adapted for 5x5 rules
- **MNIST Filtering:** Only MNIST images for digits 0-4 are used

## Usage

The usage is similar to the original project, but all puzzles and logic are for 5x5 Sudoku. See the code and config for details.

---

# The rest of this README is inherited from the original project and may reference 9x9 Sudoku. Please refer to the above section for 5x5-specific details.

## 🌟 Features

- **Template-Based Generation**: Creates puzzles using predefined templates for different difficulty levels
- **MNIST Integration**: Incorporates MNIST handwritten digits into puzzles
- **Strategy Validation**: Ensures puzzles can be solved using specific solving strategies
- **Quality Assurance**: Comprehensive validation of puzzle quality and solvability
- **Multiple Difficulty Levels**: Easy, moderate, and hard puzzles with appropriate complexity
- **Configurable Generation**: Extensive configuration options for fine-tuning puzzle generation

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages (see Installation section)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd game_projects/sudoku
```

2. Create and activate a virtual environment:
```bash
python -m venv mnist-sudoku-env
source mnist-sudoku-env/bin/activate  # On Windows: mnist-sudoku-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage

Generate validated puzzles:
```bash
python main.py --action generate_validated
```

### Available Actions

- `generate_validated`: Generate puzzles with full validation
- `analyze`: Analyze existing puzzle datasets
- `test_solver`: Test the solver on sample puzzles
- `create_sample`: Create a sample puzzle of specified difficulty

### Configuration

The project uses `config.yaml` for configuration. Key settings include:

- **Generation Settings**:
  - Number of puzzles per difficulty
  - Complexity parameters
  - Strategy weights
  - MNIST image settings

- **Validation Settings**:
  - Quality thresholds
  - Solvability checks
  - Strategy validation
  - Uniqueness verification

- **Output Settings**:
  - File formats
  - Image quality
  - Documentation options

## 🏗️ Project Structure

```
sudoku/
├── main.py                 # Main orchestration script
├── config.yaml            # Configuration file
├── template_based_generators.py  # Template-based puzzle generation
├── sudoku_validator.py    # Puzzle validation
├── sudoku_generator.py    # Core generation logic
├── puzzle_solver.py       # Puzzle solving implementation
├── dataset_analyzer.py    # Dataset analysis tools
├── config_manager.py      # Configuration management
└── strategy_kb/           # Strategy knowledge bases
    ├── sudoku_easy_strategies_kb.py
    ├── sudoku_moderate_strategies_kb.py
    └── sudoku_hard_strategies_kb.py
```

## 🧩 Puzzle Generation Process

1. **Template Selection**:
   - Selects appropriate template based on difficulty
   - Applies template pattern to maintain structure

2. **Solution Generation**:
   - Generates complete valid Sudoku solution
   - Applies template pattern
   - Verifies uniqueness and validity

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
- 35-45 filled cells
- Basic strategies (naked single, hidden single)
- Clear patterns and symmetry

### Moderate Puzzles
- 25-40 filled cells
- Intermediate strategies (naked pair, pointing pairs)
- Balanced complexity

### Hard Puzzles
- 17-30 filled cells
- Advanced strategies (x-wing, swordfish)
- Minimal clues with unique solutions

## 🔍 Validation Process

1. **Basic Validation**:
   - Grid structure
   - Number range
   - Initial validity

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
  - Includes puzzle grid, solution, and metadata
  - Strategy information

- **Images**:
  - MNIST digit puzzles
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

- MNIST dataset for digit images
- Sudoku solving strategies community
- Open source contributors

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker. 