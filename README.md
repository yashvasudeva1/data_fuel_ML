# Handling Imbalanced Datasets in Machine Learning

A comprehensive experiment demonstrating various techniques to handle class imbalance in machine learning datasets. This project compares multiple resampling and algorithmic approaches using a Logistic Regression classifier.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Techniques Explained](#techniques-explained)
- [Results](#results)
- [Dataset Information](#dataset-information)
- [Contributing](#contributing)

## Overview

Class imbalance is a common problem in real-world datasets where one class significantly outnumbers the other. This can lead to biased models that perform poorly on the minority class. This project implements and compares **6 different techniques** to address this challenge:

1. **Baseline** (No balancing)
2. **Random Oversampling**
3. **Random Undersampling**
4. **SMOTE** (Synthetic Minority Over-sampling Technique)
5. **Class Weight Adjustment**
6. **Balanced Random Forest**

## Project Structure

```
├── main.py                     # Main script to run the experiment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── Balanced_data.csv       # Balanced dataset for comparison
│   └── Imbalanced_data.csv     # Imbalanced dataset (used in experiments)
├── results/
│   ├── Comparison for Decision Tree.png
│   ├── Comparison for KNN.png
│   ├── Comparison for Logistic regression.png
│   ├── Comparison for Random Forest.png
|   ├── Comparison for SVC.png
│   └── Consolidated Comparison.png
└── techniques/
    ├── balanced_random_forest.py       # Balanced Random Forest implementation
    ├── class_weight_adjustment.py      # Class weight adjustment technique
    ├── random_oversampling.py          # Random oversampling implementation
    ├── random_undersampling.py         # Random undersampling implementation
    ├── SMOTE.py                        # SMOTE implementation
    └── stratified_train_test_split.py  # Stratified splitting utility
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd "Post 1"

# Or simply download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and utilities
- `imbalanced-learn` - Resampling techniques for imbalanced datasets

## Usage

### Running the Main Experiment

Execute the main script to run all techniques and see the comparison:

```bash
python main.py
```

### Expected Output

The script will output a comparison table showing performance metrics for each technique:

```
Performance Comparison on Imbalanced Dataset

                   Technique  Accuracy  Precision    Recall  F1 Score
0                   Baseline      0.XX       0.XX      0.XX      0.XX
1         Random Oversampling     0.XX       0.XX      0.XX      0.XX
2        Random Undersampling     0.XX       0.XX      0.XX      0.XX
3                      SMOTE      0.XX       0.XX      0.XX      0.XX
4    Class Weight Adjustment      0.XX       0.XX      0.XX      0.XX
5     Balanced Random Forest      0.XX       0.XX      0.XX      0.XX
```

### Using Individual Techniques

You can also import and use individual techniques in your own projects:

```python
# Random Oversampling
from techniques.random_oversampling import random_oversample
X_resampled, y_resampled = random_oversample(X_train, y_train)

# Random Undersampling
from techniques.random_undersampling import random_undersample
X_resampled, y_resampled = random_undersample(X_train, y_train)

# SMOTE
from techniques.SMOTE import apply_smote
X_resampled, y_resampled = apply_smote(X_train, y_train)

# Class Weight Adjustment
from techniques.class_weight_adjustment import train_with_class_weights
model = train_with_class_weights(X_train, y_train)

# Balanced Random Forest
from techniques.balanced_random_forest import train_balanced_random_forest
model = train_balanced_random_forest(X_train, y_train)

# Stratified Train-Test Split
from techniques.stratified_train_test_split import stratified_split
X_train, X_test, y_train, y_test = stratified_split(X, y)
```

## Techniques Explained

### 1. Baseline (No Balancing)
Standard Logistic Regression without any class imbalance handling. Serves as a reference point for comparison.

### 2. Random Oversampling
Randomly duplicates samples from the minority class to balance the class distribution. Simple but may lead to overfitting.

### 3. Random Undersampling
Randomly removes samples from the majority class. Fast but may lose important information.

### 4. SMOTE (Synthetic Minority Over-sampling Technique)
Creates synthetic samples for the minority class by interpolating between existing minority samples. More sophisticated than simple oversampling.

### 5. Class Weight Adjustment
Assigns higher weights to minority class samples during model training. The model penalizes misclassification of minority samples more heavily.

### 6. Balanced Random Forest
An ensemble method that combines random undersampling with Random Forest. Each tree is trained on a balanced bootstrap sample.

## Results

The `Images/` folder contains visualizations comparing technique performance across different classifiers:

| Image | Description |
|-------|-------------|
| `Comparison for Logistic regression.png` | Performance comparison using Logistic Regression |
| `Comparison for Decision Tree.png` | Performance comparison using Decision Tree |
| `Comparison for KNN.png` | Performance comparison using K-Nearest Neighbors |
| `Comparison for Random Forest.png` | Performance comparison using Random Forest |
| `Consolidated Comparison.png` | Overall comparison across all classifiers |

### Metrics Used

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of all positive predictions, how many were actually positive |
| **Recall** | Of all actual positives, how many were correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |

> **Note:** For imbalanced datasets, **Recall** and **F1 Score** are often more informative than Accuracy.

## Dataset Information

The experiment uses two datasets located in the `data/` folder:

- **Imbalanced_data.csv**: The primary dataset with significant class imbalance (used in the main experiment)
- **Balanced_data.csv**: A balanced version for comparison purposes

Both datasets should have:
- Feature columns for model training
- A `target` column containing binary class labels (0 or 1)

### Using Your Own Dataset

To use your own dataset:

1. Ensure your data is in CSV format
2. Include a column named `target` with binary labels
3. Update the path in `main.py`:

```python
X, y = load_dataset("data/your_dataset.csv")
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'imblearn'**
```bash
pip install imbalanced-learn
```

**2. ImportError with scikit-learn**
```bash
pip install --upgrade scikit-learn
```

**3. FileNotFoundError for dataset**
Ensure you're running the script from the project root directory:
```bash
cd "Post 1"
python main.py
```

## Contributing

Feel free to contribute by:
- Adding new balancing techniques
- Testing with different classifiers
- Improving documentation
- Reporting issues

## License

This project is available for educational and research purposes.

## References

- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research 16 (2002): 321-357.
