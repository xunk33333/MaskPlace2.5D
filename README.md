# MaskPlace2.5D 
Due to intellectual property restrictions, we are unable to directly provide the core code of MaskPlace2.5D. Howere this repository contains a PyTorch implementation of the compact thermal model from the paper:

**"ATPlace2.5D: Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC"**

The model provides fast thermal simulation for 2.5D integrated circuit designs with multiple chiplets. 
Thanks to ATPlace2.5D's open-source repository: https://github.com/Brilight/ATPlace_pub

## 🏗️ Architecture

### Core Components

- **`compact_themal_model.py`**: Main PyTorch model implementation
- **`Thermal.py`**: HotSpot thermal solver interface
- **`process_thermal.py`**: Data processing and thermal simulation pipeline
- **`train_compact_themal_model.py`**: Training script for the compact model
- **`plot.py`**: Visualization utilities

## 📁 Project Structure

```
atplace/
├── cases/                    # Test cases with chiplet layouts
│   ├── Case1/
│   │   ├── Case1_1.pl       # Layout file 1
│   │   ├── Case1_2.pl       # Layout file 2
│   │   ├── ...
│   │   ├── Case1.power      # Power consumption data
│   │   ├── Case1.intpsize   # Interposer size
│   │   └── Case1.blocks     # Block definitions
│   ├── Case2/
│   └── ...
├── model/                    # Trained model weights
│   ├── Case1_thermal_model.pth
│   ├── Case2_thermal_model.pth
│   └── ...
├── thermal/                  # HotSpot thermal simulator
│   ├── hotspot              # HotSpot executable
│   ├── hotspot.config       # Configuration file
│   └── new_hotspot.config   # Generated configuration
├── utils/                    # Utility functions
│   ├── blocks_parser.py     # Block definition parser
│   ├── fill_space.py        # Floorplan filler
│   ├── nets_parser.py       # Netlist parser
│   ├── pl_parser.py         # Placement file parser
│   └── uscs_parser.py       # UCS file parser
├── compact_themal_model.py  # Main thermal model
├── Thermal.py               # HotSpot interface
├── process_thermal.py       # Data processing
├── train_compact_themal_model.py  # Training script
├── train_compact_themal_model.sh  # Batch training script
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 1.8+
- NumPy
- HotSpot thermal simulator (included in `thermal/` directory)
- CUDA-enabled GPU (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/xunk33333/MaskPlace2.5D
```

2. Install required Python packages:
```bash
pip install torch numpy tqdm
```
### Prepare Dataset
```bash
python porcess_thermal.py
```
### Training the Model

To train the compact thermal model for a specific case:

```bash
python train_compact_themal_model.py --case 1 --train
```

This will:
1. Train the compact thermal model
2. Save the trained model to `model/Case1_thermal_model.pth`
3. Calculate performance metrics (MAE, RMSE, etc.) and generate comparison plots
### Batch Training

To train models for all 10 cases simultaneously:

```bash
bash train_compact_themal_model.sh
```

### Testing the Model

To only test a trained model:

```bash
python train_compact_themal_model.py --case 1
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support, please open an issue in the GitHub repository.

