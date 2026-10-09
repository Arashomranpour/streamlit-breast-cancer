<div align="center">

# 🩺 Breast Cancer Predictor

**An interactive Streamlit app that predicts whether a tumor is benign or malignant from cell-nucleus measurements, using logistic regression.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

![App screenshot](https://github.com/user-attachments/assets/cabaaa8d-fc43-4a49-b035-fddbfa590a36)

## ✨ Features

- 🎚️ **Interactive inputs** - adjust the tissue measurements with sliders in the sidebar.
- 🕸️ **Radar chart** - visualizes the measurements (Plotly).
- 🤖 **Prediction** - a Logistic Regression model (with a `StandardScaler`) classifies the sample as **Benign** or **Malicious** and shows both class probabilities.
- 🎨 Custom CSS styling for the diagnosis badge.

Uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.

> ⚠️ This app is a learning project that can assist exploration. It is **not** a substitute for professional medical diagnosis.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `data.csv` - the Breast Cancer Wisconsin dataset (columns `id`, `diagnosis`, the 30 measurement features)

### Install

```bash
git clone https://github.com/Arashomranpour/streamlit-breast-cancer.git
cd streamlit-breast-cancer
pip install streamlit pandas numpy scikit-learn plotly
```

### Train the model, then run the app

```bash
# 1. Train: creates model.pkl and scaler.pkl (also prints accuracy + classification report)
python src/main.py

# 2. Launch the app
streamlit run src/stream.py
```

Keep `data.csv`, `style.css`, `model.pkl` and `scaler.pkl` in the folder you run the commands from, as the scripts load them by relative path.

## 📁 Project Structure

```
.
├── src/
│   ├── main.py      # Data cleaning, training, evaluation, saves model + scaler
│   └── stream.py    # Streamlit app: sliders, radar chart, prediction
└── style.css        # App styling
```

## 🛠️ Tech Stack

`Streamlit` · `scikit-learn` · `pandas` · `NumPy` · `Plotly`

## 🤝 Contributing

Pull requests and issues are welcome.
