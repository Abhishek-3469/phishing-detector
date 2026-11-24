import os
import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_URL = "https://raw.githubusercontent.com/Khiladi-786/Phishing_Deployment/main/dataset_phishing.csv"

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "signin", "paypal", "confirm", "free", "support", "webscr", "ebayisapi"
]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"]
RISKY_TLDS = [".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".ru", ".cn", ".buzz", ".work", ".info"]

def calc_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def extract_features(url):
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    path = parsed.path
    lower_u = u.lower()
    clean_u = re.sub(r"^https?://", "", lower_u)

    digits = sum(ch.isdigit() for ch in u)
    letters = sum(ch.isalpha() for ch in u)
    length = max(len(u), 1)

    return {
        "clean_url": clean_u,
        "url_len": len(u),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(parsed.query),
        "dots": u.count("."),
        "hyphens": u.count("-"),
        "slashes": u.count("/"),
        "digits": digits,
        "digit_ratio": digits / length,
        "letter_ratio": letters / length,
        "entropy": calc_entropy(clean_u),
        "has_at": 1 if "@" in u else 0,
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", host) else 0,
        "is_https": 1 if parsed.scheme == "https" else 0,
        "double_slash_path": 1 if "//" in path else 0,
        "subdomains": max(host.count(".") - 1, 0),
        "suspicious_words": sum(w in lower_u for w in SUSPICIOUS_WORDS),
        "has_shortener": 1 if any(s in host for s in SHORTENERS) else 0,
        "risky_tld": 1 if any(host.endswith(t) for t in RISKY_TLDS) else 0,
        "special_chars": sum(u.count(c) for c in ["?", "=", "&", "%", "_"]),
    }

@st.cache_resource
def load_and_train():
    source = "urls.csv" if os.path.exists("urls.csv") else DATA_URL
    df = pd.read_csv(source)[["url", "status"]]
    df = df.dropna().drop_duplicates(subset="url").copy()
    df["url"] = df["url"].str.strip()
    df["label"] = df["status"].map({"legitimate": 0, "phishing": 1})
    df = df.dropna(subset=["label"])

    X = pd.DataFrame([extract_features(u) for u in df["url"]])
    y = df["label"].astype(int)
    num_cols = [c for c in X.columns if c != "clean_url"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = Pipeline([
        ("prep", ColumnTransformer([("num", "passthrough", num_cols)])),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    hybrid_model = Pipeline([
        ("prep", ColumnTransformer([
            ("tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5),
                min_df=2, max_features=30000, sublinear_tf=True
            ), "clean_url"),
            ("num", StandardScaler(), num_cols),
        ])),
        ("clf", LogisticRegression(C=3.0, max_iter=1000)),
    ])

    model = VotingClassifier(
        estimators=[("hybrid", hybrid_model), ("rf", rf_model)],
        voting="soft",
        weights=[2, 1],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    return model, metrics

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")
st.title("Phishing URL Detector")
st.write("Paste any website link below to check whether it looks legitimate or suspicious.")

with st.spinner("Loading phishing detection model..."):
    model, metrics = load_and_train()

with st.sidebar:
    st.header("Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("Precision", f"{metrics['precision']*100:.2f}%")
    st.metric("Recall", f"{metrics['recall']*100:.2f}%")
    st.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
    st.caption("Ensemble: Char N-gram Logistic Regression + Random Forest")

url_input = st.text_input("Enter URL", placeholder="https://example.com/login")

if st.button("Check URL"):
    if not url_input.strip():
        st.warning("Please enter a URL first.")
    else:
        feats = extract_features(url_input)
        row = pd.DataFrame([feats])
        pred = model.predict(row)[0]
        risk = model.predict_proba(row)[0][1] * 100

        if pred == 1:
            st.error(f"Phishing Detected (Risk Score: {risk:.1f}%)")
        else:
            st.success(f"Likely Legitimate (Phishing Risk: {risk:.1f}%)")

        flags = []
        if feats["has_ip"]:
            flags.append("Uses a raw IP address instead of a domain name")
        if feats["has_at"]:
            flags.append("Contains an '@' symbol")
        if not feats["is_https"]:
            flags.append("Does not use HTTPS")
        if feats["double_slash_path"]:
            flags.append("Contains an extra '//' redirect pattern in the path")
        if feats["suspicious_words"] > 0:
            flags.append(f"Contains {feats['suspicious_words']} sensitive keyword(s)")
        if feats["has_shortener"]:
            flags.append("Uses a URL shortener")
        if feats["risky_tld"]:
            flags.append("Uses a commonly abused top-level domain")
        if feats["subdomains"] >= 3:
            flags.append("Has an unusually high number of subdomains")
        if feats["entropy"] > 4.2:
            flags.append("High character randomness in URL")

        if flags:
            st.subheader("Risk Indicators Found")
            for item in flags:
                st.write(f"- {item}")
        else:
            st.info("No common red flags spotted in the URL structure.")