# Sentiment360

Sentiment360 is a machine learning project focused on Aspect-Based Sentiment Analysis (ABSA) for laptop reviews. It leverages the **SemEval-2014 Task 4 Laptop Reviews** dataset and explores different modeling techniques ranging from a simple Machine Learning baseline (Logistic Regression) to Deep Learning approaches (LSTM) built with PyTorch.

## Data Source

The dataset used in this project is downloaded from Hugging Face: `tomaarsen/setfit-absa-semeval-laptops`. 
The dataset consists of laptop reviews categorized into 4 sentiment classes:
- `0`: Negative
- `1`: Neutral
- `2`: Positive
- `3`: Conflict

## Project Structure

```
Sentiment360/
│
├── data/                         # Directory containing downloaded datasets (CSV)
│   ├── laptop_train.csv          # Training dataset
│   └── laptop_test.csv           # Testing dataset
│
├── data_exploration.py           # Script to download and format the data from Hugging Face
├── ml_baseline.py                # Machine learning baseline (TF-IDF + Logistic Regression)
├── lstm_model.ipynb              # Jupyter Notebook implementing a PyTorch LSTM model
├── laptop_sentiment_lstm.pth     # Saved PyTorch model weights for the LSTM
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup and Installation

1. Clone or navigate to the project directory.
2. Ensure you have Python installed.
3. Install the dependencies (you may need to install `scikit-learn` and `torch` in addition to what is provided):
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn torch jupyter
   ```

## How to Run

### 1. Data Preparation
To download and prepare the dataset, run:
```bash
python data_exploration.py
```
This will fetch the dataset from Hugging Face and save `laptop_train.csv` and `laptop_test.csv` in the `data/` folder.

### 2. Machine Learning Baseline
To run the Logistic Regression baseline model with TF-IDF features:
```bash
python ml_baseline.py
```
This script will train the model, evaluate its accuracy on the test set, and print a detailed classification report.

### 3. Deep Learning LSTM Model
To explore and run the PyTorch LSTM model, open the Jupyter Notebook:
```bash
jupyter notebook lstm_model.ipynb
```
The notebook contains step-by-step instructions for data preprocessing, building the LSTM architecture, training, and evaluation. The trained model weights are saved in `laptop_sentiment_lstm.pth`.

## Model Architectures

1. **Baseline Model:**
   - **Feature Extraction:** TF-IDF Vectorizer (max 5000 features)
   - **Classifier:** Logistic Regression (max_iter=1000)
2. **LSTM Model:**
   - **Framework:** PyTorch
   - **Architecture:** Embedding Layer -> 2-Layer LSTM (Hidden Dim: 256) -> Fully Connected Linear Layer
   - **Output:** 4 Sentiment Classes
