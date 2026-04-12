# Social Media Sentiment Analysis

A sentiment analysis project for short social-media style text using Python, NLTK, scikit-learn, and Streamlit. The repository includes preprocessing notebooks, a trained TF-IDF plus Naive Bayes model, and a web app for interactive prediction.

## Features

- Multiclass prediction: `negative`, `neutral`, `positive`
- Text preprocessing with URL removal, tokenization, stopword removal, and lemmatization
- Saved TF-IDF vectorizer and trained classifier in `models/`
- Streamlit app for single-text and bulk prediction
- VADER lexicon fallback for phrases the trained model does not understand well
- Notebook workflow for preprocessing, feature extraction, training, and evaluation

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

## Tech Stack

- Python 3.14
- Pandas, NumPy
- NLTK
- scikit-learn
- Streamlit
- Joblib
- Matplotlib, Seaborn, WordCloud

## Setup

### 1. Create a virtual environment

From the project root:

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

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

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. If `Activate.ps1` is missing

Rebuild the environment:

```powershell
python -m venv --clear .venv
```

This project previously hit a partial `venv` creation issue where `pip` was installed but the activation scripts were missing. Recreating the environment fixes that.

### 5. Install dependencies

After activation:

```powershell
pip install -r requirements.txt
```

## Run the App

Start the Streamlit app from the project root:

```powershell
streamlit run app.py
```

The app uses:

- `models/tfidf_vectorizer.pkl`
- `models/best_model.pkl`
- `models/vader_lexicon.txt`

## How Prediction Works

1. Raw text is cleaned using `text_preprocessing.py`
2. Cleaned text is transformed using the saved TF-IDF vectorizer
3. The trained model predicts one of the three sentiment classes
4. If the vectorizer finds no useful features, or the model confidence is weak, the project falls back to a local VADER lexicon score

This improved the behavior for common real-world phrases such as:

- `i like this product`
- `I love this product`
- `This is terrible`
- `It is okay`

## Notebook Workflow

Run the notebooks in this order to rebuild the pipeline:

1. `notebooks/sentiment_analysis.ipynb`
2. `notebooks/preprocessing.ipynb`
3. `notebooks/feature_extraction.ipynb`
4. `notebooks/model_training.ipynb`
5. `notebooks/evaluation_error_analysis.ipynb`
6. `notebooks/final_demo.ipynb`

## Data Files

- `data/sample_dataset.csv`: source dataset used in the notebook pipeline
- `data/processed_dataset.csv`: cleaned dataset after preprocessing
- `data/classification_report.txt`: saved evaluation metrics

## Model Files

- `models/tfidf_vectorizer.pkl`: fitted TF-IDF vectorizer
- `models/best_model.pkl`: trained sentiment classifier
- `models/vader_lexicon.txt`: local copy of the VADER sentiment lexicon used as fallback support

## Known Limitations

- The current dataset is synthetic and repetitive, so reported notebook accuracy may look better than real-world performance
- A lexicon fallback helps, but it does not replace retraining on a stronger dataset
- Some NLTK resources must exist locally for preprocessing to work smoothly in offline environments

## Example Usage

```python
from predict_sentiment import predict_sentiment

print(predict_sentiment("I like this product"))
print(predict_sentiment("This is terrible"))
print(predict_sentiment("It is okay"))
```

## Troubleshooting

### `Activate.ps1` is not recognized

Use a relative path:

```powershell
.\Activate.ps1
```

or

```powershell
.\.venv\Scripts\Activate.ps1
```

Typing only `Activate.ps1` makes PowerShell search for a command, not a local script.

### `Activate.ps1` still cannot be found

Check whether the file exists:

```powershell
Test-Path .\.venv\Scripts\Activate.ps1
```

If it returns `False`, rebuild the environment:

```powershell
python -m venv --clear .venv
```

### Positive text is predicted as negative

The original model was trained on a small synthetic dataset, so it does not generalize well. The project now uses a VADER lexicon fallback to improve predictions for unseen social-media phrases.

## Future Improvements

- Retrain on a larger and more realistic social-media dataset
- Add confidence scores to the Streamlit UI
- Add tests for preprocessing and prediction behavior
- Build a more reproducible training pipeline
