# Classic Logic Puzzle Games

[![GitHub](https://img.shields.io/github/license/yourusername/classic-games)](https://github.com/yourusername/classic-games/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.20%2B-yellowgreen)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.2%2B-yellowgreen)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.4%2B-orange)](https://matplotlib.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-red)](https://opencv.org/)

A collection of three classic logic puzzle games implemented in Python, featuring sophisticated puzzle generation, validation, and solving capabilities.

## 🎮 Games Included

### 1. Sudoku
- MNIST digit integration
- Template-based generation
- Multiple difficulty levels
- Strategy-based solving
[More details](sudoku/README.md)

### 2. KenKen
- Arithmetic cage generation
- Multiple operation types
- Difficulty scaling
- Unique solution validation
[More details](kenken/README.md)

### 3. Futoshiki
- Inequality constraint generation
- Size-variable puzzles
- Difficulty progression
- Constraint validation
[More details](futoshiki/README.md)

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages (see Installation section)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd game_projects
```

2. Create and activate a virtual environment:
```bash
python -m venv classic-games-env
source classic-games-env/bin/activate  # On Windows: classic-games-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Project Structure

```
game_projects/
├── sudoku/              # Sudoku puzzle game
│   ├── main.py
│   ├── config.yaml
│   └── ...
├── kenken/             # KenKen puzzle game
│   ├── main.py
│   ├── config.yaml
│   └── ...
├── futoshiki/          # Futoshiki puzzle game
│   ├── main.py
│   ├── config.yaml
│   └── ...
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## 🎮 Running the Games

Each game can be run independently:

### Sudoku
```bash
cd sudoku
python main.py --action generate_validated
```

### KenKen
```bash
cd kenken
python kenken_main.py --action generate
```

### Futoshiki
```bash
cd futoshiki
python futoshiki_main.py --action generate
```

## 🛠️ Common Features

All games share these common features:
- Puzzle generation
- Solution validation
- Difficulty levels
- Quality checks
- Output in multiple formats
- Analysis tools

## 📊 Output Formats

Each game supports:
- JSON puzzle files
- Image exports
- Analysis reports
- Solution files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MNIST dataset for Sudoku digits
- Puzzle game communities
- Open source contributors

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker. 