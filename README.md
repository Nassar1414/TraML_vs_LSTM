
# Comparative Evaluation of Traditional ML vs. LSTM for Natural Disaster Forecasting

This repository contains code, data references, and documentation for evaluating classical machine-learning models against a sequence-based LSTM model on a synthetic natural-disaster forecasting task.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)  
4. [Model Pipelines](#model-pipelines)  
   - [Traditional ML (TraML)](#traditional-ml-traml)  
   - [LSTM Model (Final_LSTM)](#lstm-model-final_lstm)  
5. [Results Summary](#results-summary)  
6. [Getting Started](#getting-started)  
   1. [Prerequisites](#prerequisites)  
   2. [Installation](#installation)  
   3. [Usage](#usage)  
7. [Project Structure](#project-structure)  
8. [Future Work](#future-work)  
9. [Acknowledgments](#acknowledgments)  
10. [License](#license)  

---

## Project Overview

Natural disasters pose severe societal and economic impacts. Forecasting their occurrence is challenging due to:

- **Class imbalance**: disasters are rare events.  
- **Temporal dependencies**: past sequences inform future events.

This project compares:

1. **Traditional ML** pipelines (Decision Tree, Random Forest, XGBoost) with SMOTE + undersampling  
2. **Sequence-based LSTM** model with Gaussian-noise augmentation  

Performance is evaluated using accuracy, precision, recall, and F₁-score under 5-fold cross-validation.

---

## Dataset

- **Source:** Kaggle “Forecasting Disaster Management in 2024” (synthetic)  
- **Size:** ~10 000 records  
- **Key features:**  
  - `Disaster_Type`, `Location` (categorical)  
  - `Magnitude`, `Fatalities`, `Economic_Loss` (numerical)  
  - `Date` (decomposed into Year/Month/Day/Day_of_Week/Season)  
- **Target:** Binary `Disaster_Occurred` (1 = disaster; 0 = no-disaster, with 30 % artificial no-disaster injection)

---

## Preprocessing & Feature Engineering

- **Missing data:** Dropped incomplete rows to preserve sequence integrity  
- **Date features:** Extracted Year, Month, Day, Day_of_Week, Season  
- **Encoding & Scaling:**  
  - Label-encoded categorical features  
  - Min-Max scaled numerical features  
- **Feature selection (TraML):** Mutual-information scores → top 5 features  
- **Balancing:**  
  - TraML: SMOTE + random undersampling  
  - LSTM: Gaussian-noise augmentation on non-disaster periods  

---

## Model Pipelines

### Traditional ML (TraML)

- **Algorithms:**  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- **Workflow:**  
  1. Load and preprocess data  
  2. Balance with SMOTE → undersample  
  3. 5-fold stratified cross-validation  
  4. Compute accuracy, precision, recall, F₁  
- **Implementation:** [`traml.py`](traml.py)

### LSTM Model (Final_LSTM)

- **Architecture:** Stacked LSTM layers with Dropout & BatchNorm  
- **Sequence prep:** Sliding window of 30 time-steps  
- **Augmentation:** Gaussian noise injected into non-disaster segments  
- **Training:**  
  - EarlyStopping callback  
  - Min-Max scaling per feature  
- **Evaluation:** 5-fold CV on flattened sequences  
- **Implementation:** [`final_lstm.py`](final_lstm.py)

---

## Results Summary

| Model              | Accuracy   | Precision   | Recall      | F₁ Score   |
| ------------------ | ---------- | ----------- | ----------- | ---------- |
| Decision Tree      | ~77 %      | —           | —           | ~0.76      |
| Random Forest      | ~86 %      | —           | —           | ~0.84      |
| **XGBoost**        | ~89 %      | —           | —           | ~0.86      |
| **LSTM (final)**   | **99.92 %**| **99.85 %** | **100.00 %**| **99.92 %**|

> *All metrics based on 5-fold cross-validation.*  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Access to the synthetic dataset CSV (`natural_disasters_2024.csv`)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# (Optional) Create and activate a venv
python3 -m venv venv
source venv/bin/activate

# Install common dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost tensorflow
```

### Usage

1. **Traditional ML pipeline**  
   ```bash
   python traml.py      --data-path path/to/natural_disasters_2024.csv      --output-dir results/traml
   ```
2. **LSTM pipeline**  
   ```bash
   python final_lstm.py      --data-path path/to/natural_disasters_2024.csv      --sequence-length 30      --output-dir results/lstm
   ```

Each script logs cross-validation metrics and saves trained models & evaluation reports under `results/`.

---

## Project Structure

```
├── data/
│   └── natural_disasters_2024.csv      # Synthetic dataset
├── final_lstm.py                       # LSTM training & evaluation
├── traml.py                            # Traditional ML pipeline
├── Report_Phase_2.docx                # Full project report
├── Research.pptx                       # Slide deck
├── requirements.txt                    # (Optional) pin dependencies
└── README.md                           # This file
```

---

## Future Work

- Validate models on real-world EM-DAT and NOAA disaster records  
- Extend to multi-class/multi-output forecasting of disaster types  
- Incorporate unstructured data (news, social media) via NLP (BERT, GPT)  
- Explore Transformer-based and hybrid architectures with explainability (SHAP, LIME)  
- Deploy real-time geospatial dashboards and REST APIs  

---

## Acknowledgments

- Synthetic dataset provided by the “Forecasting Disaster Management in 2024” Kaggle competition  
- Report and methodology by the project team (see `Report_Phase_2.docx`)  

---

## License

This project is released under the [MIT License](LICENSE) (or your chosen license).
