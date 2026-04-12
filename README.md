# Social Media Sentiment Analysis

A machine learning project for classifying short social-media text as **negative**, **neutral**, or **positive**. The repository combines text preprocessing, TF-IDF feature extraction, a Naive Bayes classifier, and a Streamlit interface for interactive sentiment prediction.

Local app URL after starting Streamlit: [http://localhost:8501](http://localhost:8501)

Hosted Streamlit app: [Open the Streamlit project](https://socialmediaandsentimentanalysis.streamlit.app/)

## Overview

This project was built to explore a complete sentiment analysis workflow, from raw text preprocessing and feature engineering to model training, evaluation, and deployment in a lightweight web application.

To improve prediction quality on unseen real-world phrases, the project also includes a local VADER lexicon fallback for cases where the trained classifier has limited vocabulary coverage.

## Key Features

- Multiclass sentiment prediction: `negative`, `neutral`, `positive`
- Text preprocessing with URL removal, tokenization, stopword removal, and lemmatization
- TF-IDF vectorization with a trained Naive Bayes classifier
- Streamlit web app for single-text and bulk prediction
- Notebook-based workflow for preprocessing, feature extraction, training, and evaluation
- Lexicon-based fallback using VADER for better handling of unseen phrases

## Project Structure

```text
.
|- app.py
|- predict_sentiment.py
|- text_preprocessing.py
|- requirements.txt
|- README.md
|- data/
|  |- sample_dataset.csv
|  |- processed_dataset.csv
|  `- classification_report.txt
|- models/
|  |- best_model.pkl
|  |- tfidf_vectorizer.pkl
|  `- vader_lexicon.txt
`- notebooks/
   |- sentiment_analysis.ipynb
   |- preprocessing.ipynb
   |- feature_extraction.ipynb
   |- model_training.ipynb
   |- evaluation_error_analysis.ipynb
   `- final_demo.ipynb
```

## Technology Stack

- Python 3.14
- Pandas and NumPy
- NLTK
- scikit-learn
- Streamlit
- Joblib
- Matplotlib, Seaborn, and WordCloud

## Setup

### 1. Create a virtual environment

From the project root:

```powershell
python -m venv .venv
```

### 2. Activate the environment

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you are already inside `.venv\Scripts`, use:

```powershell
.\Activate.ps1
```

### 3. If PowerShell blocks script execution

Run this once in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. If `Activate.ps1` is missing

Rebuild the environment:

```powershell
python -m venv --clear .venv
```

This resolves cases where the virtual environment is only partially created and the activation scripts are missing.

### 5. Install dependencies

After activation:

```powershell
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app from the project root:

```powershell
streamlit run app.py
```

The application uses the following artifacts:

- `models/tfidf_vectorizer.pkl`
- `models/best_model.pkl`
- `models/vader_lexicon.txt`

## Prediction Pipeline

1. Input text is cleaned using the preprocessing utilities in `text_preprocessing.py`
2. The cleaned text is transformed using the saved TF-IDF vectorizer
3. The trained classifier predicts a sentiment label
4. If the vectorizer finds no useful features, or the classifier confidence is weak, the system falls back to a VADER-based lexicon score

This hybrid approach improves behavior for phrases such as:

- `i like this product`
- `I love this product`
- `This is terrible`
- `It is okay`

## Notebook Workflow

Run the notebooks in the following order to reproduce the pipeline:

1. `notebooks/sentiment_analysis.ipynb`
2. `notebooks/preprocessing.ipynb`
3. `notebooks/feature_extraction.ipynb`
4. `notebooks/model_training.ipynb`
5. `notebooks/evaluation_error_analysis.ipynb`
6. `notebooks/final_demo.ipynb`

All notebooks were executed successfully during project verification.

## Data and Model Artifacts

### Data Files

- `data/sample_dataset.csv`: source dataset used in the workflow
- `data/processed_dataset.csv`: cleaned dataset generated after preprocessing
- `data/classification_report.txt`: saved evaluation metrics

### Model Files

- `models/tfidf_vectorizer.pkl`: fitted TF-IDF vectorizer
- `models/best_model.pkl`: trained sentiment classifier
- `models/vader_lexicon.txt`: local VADER lexicon used as fallback support

## Example Usage

```python
from predict_sentiment import predict_sentiment

print(predict_sentiment("I like this product"))
print(predict_sentiment("This is terrible"))
print(predict_sentiment("It is okay"))
```

## Limitations

- The current dataset is synthetic and repetitive, so reported notebook accuracy may be higher than real-world performance
- The lexicon fallback improves robustness, but it does not replace retraining on a larger and more realistic dataset
- Some NLTK resources must be available locally for preprocessing to work smoothly in offline environments

## Troubleshooting

### `Activate.ps1` is not recognized

Use a relative path instead of typing only the script name:

```powershell
.\Activate.ps1
```

or

```powershell
.\.venv\Scripts\Activate.ps1
```

### `Activate.ps1` cannot be found

Check whether the file exists:

```powershell
Test-Path .\.venv\Scripts\Activate.ps1
```

If it returns `False`, rebuild the environment:

```powershell
python -m venv --clear .venv
```

### Positive text is predicted as negative

The original classifier was trained on a limited synthetic dataset, so it does not generalize perfectly. The VADER fallback was added to improve prediction quality on unseen social-media phrases.

## Future Improvements

- Retrain on a larger and more realistic social-media dataset
- Add confidence scores to the Streamlit interface
- Add automated tests for preprocessing and prediction behavior
- Build a more reproducible training and evaluation pipeline
