# Phishing URL Detector

A lightweight machine learning web app that checks whether a URL looks legitimate or suspicious before you click it. It flags risky links with **93.92% accuracy** and **92.74% recall** and highlights the specific red flags found in the URL structure.

**Live App:** https://phishing-detector-hukfakkgszrcvwqwcpkl4a.streamlit.app/

## Problem
Everyday users and small teams often cannot tell disguised phishing links apart from legitimate websites at a glance. This tool analyzes raw URL text and structural patterns to estimate phishing risk in real time without relying on external paid APIs.

## How It Works
1. **Data Cleaning:** Uses a balanced dataset of 11,430 labeled URLs (11,429 usable entries after stripping whitespace, nulls, and duplicates).
2. **Feature Engineering:** Extracts 20 lexical signals from each URL—including length, subdomain count, raw IP usage, HTTPS check, suspicious keywords, URL shorteners, commonly abused TLDs, and Shannon character entropy—alongside character n-grams (2–5 chars) via TF-IDF.
3. **Model:** Soft-voting ensemble combining a TF-IDF + Logistic Regression pipeline (weight 2) with a 200-tree Random Forest classifier (weight 1).
4. **Interface:** Built with Streamlit to display the verdict, risk percentage, and human-readable risk indicators.

## Model Performance
Evaluated on a stratified 80/20 train-test split:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 76.90% | 79.71% | 72.18% | 75.76% |
| Random Forest (Lexical Only) | 86.79% | 86.66% | 86.96% | 86.81% |
| **Final Ensemble (Hybrid + RF)** | **93.92%** | **94.98%** | **92.74%** | **93.85%** |

## Sample Outputs
- `https://www.wikipedia.org/wiki/Cybersecurity` → **Likely Legitimate** (Phishing Risk: 4.0%)
- `http://secure-login-paypal-verify.account-update.com/signin` → **Phishing Detected** (Risk Score: 94.2% — no HTTPS, 7 sensitive keywords, high character randomness)

## Project Files
- `phishing_detector.py` — Complete self-contained script (data loading, feature extraction, model training, and Streamlit UI).
- `urls.csv` — Labeled URL dataset (automatically fetched online if missing locally).
- `requirements.txt` — Python dependencies.

## Run Locally
```bash
git clone [https://github.com/Abhishek-3469/phishing-detector.git](https://github.com/Abhishek-3469/phishing-detector.git)
cd phishing-detector
pip install -r requirements.txt
streamlit run phishing_detector.py