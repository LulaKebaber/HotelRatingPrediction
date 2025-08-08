# Hotel Review Score Prediction

This project focuses on predicting hotel review scores using a variety of features extracted from a real-world dataset. The primary goal is to build a regression model that estimates the `Review_Score` based on textual and non-textual data. The project emphasizes data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation.

## 📊 Dataset

The dataset is sourced from:

https://frasca.di.unimi.it/MLDNN/input_data.csv


Each row in the dataset represents a hotel review with attributes including:

- Text reviews (`Review`)
- Hotel metadata (`Hotel_Name`, `Hotel_Address`)
- Temporal features (`Review_Date`)
- Categorical features (`Reviewer_Nationality`, `Review_Type`, etc.)
- Target variable: `Review_Score`

> **Note:** For faster training and experimentation, only a 3% sample of the full dataset is used in this notebook.

## 🔍 Project Structure

The notebook is divided into the following main sections:

### 1. Dataset Overview
- Overview of data shape and missing values.
- Distribution analysis of review scores and review lengths.

### 2. Data Preprocessing
- Dropping irrelevant or high-cardinality features (`Hotel_Address`, `Review_Date`, etc.).
- Label encoding for categorical features.
- Tokenization and padding of textual reviews.
- Integration of pre-trained GloVe embeddings for semantic understanding.

### 3. Model Architecture: BiLSTM with Attention

The main model is built using **TensorFlow** and includes:

- **Embedding layer** initialized with **pre-trained GloVe embeddings** (300D).
- **Bidirectional LSTM (BiLSTM)** to capture sequential dependencies from both directions.
- **Attention mechanism** to focus on the most relevant parts of each review.
- **Dense layers** for final regression output.

### 4. Training & Evaluation

The model is trained as a **regression model** with the following settings:

- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**:
  - MSE (Mean Squared Error)
  - R² Score (Coefficient of Determination)

Model performance is tracked on a validation set and visualized via learning curves.

## Technologies Used

- Python 3
- pandas, numpy
- seaborn, matplotlib
- TensorFlow / Keras
- GloVe word embeddings
- scikit-learn (for metrics and preprocessing support)

## 📄 License

This project is for educational and academic purposes. You may adapt and build upon it with attribution.
