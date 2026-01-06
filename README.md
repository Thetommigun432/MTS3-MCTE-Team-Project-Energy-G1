# Team Project - NILM (Non-Intrusive Load Monitoring)

Project for energy consumption recognition and disaggregation using Deep Learning techniques (LSTM).

## Project Structure

- `clean_excel.py`: Cleaning and preparing raw data from InfluxDB
- `explore_data.py`: Exploratory data analysis (EDA)
- `prepare_data.py`: Data preparation for training (temporal alignment, sequence creation)
- `nilm_model.py`: LSTM model definition for NILM
- `train_nilm.py`: Script to train models
- `predict_nilm.py`: Script to make predictions and evaluate models

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Data Cleaning
```bash
python clean_excel.py
```

### 2. Data Exploration
```bash
python explore_data.py
```

### 3. Data Preparation for Training
This script:
- Temporally aligns total consumption with individual appliance consumption
- Creates sequences using sliding window (default: 60 timesteps)
- Normalizes data
- Splits into train/test sets

```bash
python prepare_data.py
```

### 4. Model Training
Trains a separate LSTM model for each appliance:

```bash
python train_nilm.py
```

Trained models are saved in `models/`.

### 5. Evaluation and Predictions
Evaluates models on test set and displays metrics:

```bash
python predict_nilm.py
```

## Model Architecture

The project uses a **Sequence-to-Point** approach with LSTM:

- **Input**: Total consumption sequence (60 timesteps)
- **Output**: Predicted consumption of a single appliance
- **Strategy**: One separate model for each main appliance

### LSTM Model

The model includes:
- LSTM layers to capture temporal patterns
- Dense layers for final regression
- Dropout for regularization
- Linear output (regression)

## Notes

- Data must be in `data/influxdb_query_20251020_074134_cleaned.xlsx`
- The model requires at least 1000 samples per appliance
- Sequence length can be modified in `prepare_data.py` (default: 60)