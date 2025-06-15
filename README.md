# Classic Games Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-blue)](https://matplotlib.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-blue)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.18%2B-blue)](https://scikit-image.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-blue)](https://scikit-learn.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.62%2B-blue)](https://tqdm.github.io/)
[![PyYAML](https://img.shields.io/badge/PyYAML-5.4%2B-blue)](https://pyyaml.org/)

A collection of classic logic puzzle games implemented with modern AI and computer vision techniques. This project is developed by the [Department of Sustainability and Innovation in Digital Ecosystems](https://www.sust.wiwi.uni-due.de) at the University of Duisburg-Essen.

## Overview

This project implements three classic logic puzzle games:
1. Sudoku with MNIST digit recognition
2. KenKen with arithmetic operation recognition
3. Futoshiki with inequality constraint recognition

Each game features:
- Computer vision-based input processing
- AI-powered puzzle generation
- Multiple difficulty levels
- Strategy-based solving approaches
- Performance analytics

## 🎮 Games Included

### 1. Sudoku
- MNIST digit recognition
- Multiple difficulty levels
- Strategy-based solving
- Performance tracking

### 2. KenKen
- Arithmetic operation recognition
- Multiple grid sizes
- Strategy-based solving
- Performance tracking

### 3. Futoshiki
- Inequality constraint recognition
- Multiple grid sizes
- Strategy-based solving
- Performance tracking

## 🚀 Getting Started

### Quick Setup
For a quick setup of the development environment, run:
```bash
python unified_env_quick_setup.py
```
This script will:
- Create a virtual environment
- Install all required dependencies
- Set up the project structure
- Download necessary datasets

### Manual Setup
If you prefer to set up manually:

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

#### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/classic-games.git
cd classic-games
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
classic-games/
├── game_projects/
│   ├── sudoku/
│   ├── kenken/
│   └── futoshiki/
├── requirements.txt
└── README.md
```

## 🎯 Running the Games

### Sudoku
```bash
cd game_projects/sudoku
python main.py --action generate_validated
```

### KenKen
```bash
cd game_projects/kenken
python main.py --action generate_validated
```

### Futoshiki
```bash
cd game_projects/futoshiki
python main.py --action generate_validated
```

## 🔍 Common Features

### 1. Puzzle Generation
- Template-based generation
- Difficulty-based generation
- Strategy-based generation
- Compositionality validation

### 2. Input Processing
- Image preprocessing
- Feature extraction
- Pattern recognition
- Validation

### 3. Solving Strategies
- Basic strategies
- Advanced strategies
- Strategy composition
- Performance tracking

### 4. Output Formats
- JSON for puzzle data
- CSV for performance metrics
- PNG for visualizations
- PDF for reports

## 🤝 Contributing

We welcome contributions to expand our collection of classic games and puzzles! This project is specifically designed to test and demonstrate compositionality in puzzle-solving strategies and knowledge bases. If you're interested in contributing, here's how you can help:

### Adding New Games
We're particularly interested in games that:
- Have clear, composable solving strategies
- Can be represented with a knowledge base
- Support multiple difficulty levels
- Can be validated for compositionality

### Implementation Guidelines
When adding a new game, please ensure:
1. **Strategy Compositionality**
   - Define basic and advanced strategies
   - Implement strategy validation
   - Ensure strategies can be composed
   - Include strategy performance tracking

2. **Knowledge Base Structure**
   - Create a clear knowledge hierarchy
   - Define strategy prerequisites
   - Implement validation rules
   - Include difficulty metrics

3. **Template System**
   - Design template patterns
   - Implement symmetry validation
   - Include difficulty ratings
   - Support strategy requirements

4. **Validation System**
   - Implement compositionality checks
   - Validate strategy sequences
   - Verify puzzle uniqueness
   - Track performance metrics

### How to Contribute
1. Fork the repository
2. Create a new branch for your game
3. Implement the game following our structure
4. Add comprehensive tests
5. Submit a pull request

### Example Game Structure
```
game_projects/your_game/
├── main.py
├── game_generator.py
├── game_validator.py
├── knowledge_base/
│   ├── basic_strategies.py
│   ├── advanced_strategies.py
│   └── strategy_validator.py
├── templates/
│   ├── easy_templates.py
│   ├── moderate_templates.py
│   └── hard_templates.py
└── tests/
    ├── test_generator.py
    ├── test_validator.py
    └── test_strategies.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The project is developed and maintained by the Department of Sustainability and Innovation in Digital Ecosystems at the University of Duisburg-Essen.

## 🙏 Acknowledgments

- Department of Sustainability and Innovation in Digital Ecosystems, University of Duisburg-Essen
- MNIST dataset for digit recognition
- OpenCV and scikit-image communities for computer vision tools
- NumPy and Pandas for data processing
- Matplotlib for visualization

## 📞 Support

For questions, suggestions, or collaboration opportunities, please contact the Department of Sustainability and Innovation in Digital Ecosystems at the University of Duisburg-Essen. 