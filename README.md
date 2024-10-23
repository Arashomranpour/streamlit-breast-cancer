# Streamlit Breast Cancer Predictor

This repository contains a web application built using **Streamlit** to predict breast cancer diagnosis using a logistic regression model. The app uses features from the **Breast Cancer Wisconsin Dataset** to determine if a tumor is benign or malignant.
![image](https://github.com/user-attachments/assets/cabaaa8d-fc43-4a49-b035-fddbfa590a36)

## Features:
- **Interactive Inputs**: Users can input breast tissue measurements via sliders in the sidebar.
- **Radar Charts**: Visualizes the data in an easy-to-understand radar chart format.
- **Prediction**: Uses a trained Logistic Regression model to predict whether the tumor is benign or malignant, providing probability estimates.

## Installation

### Requirements:
- Python 3.8+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Plotly

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/Arashomranpour/streamlit-breast-cancer.git
    cd streamlit-breast-cancer
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:
    ```bash
    python src/main.py
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run src/stream.py
    ```

   ** put all of the files in a folder **

## Data
The app uses the **Breast Cancer Wisconsin Dataset**. Ensure that the dataset (`data.csv`) is located in the `data/` folder. The dataset should have the following features:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

## Usage

- **Model Training**:  
  Run the `main.py` file to train the Logistic Regression model, which will output `model.pkl` and `scaler.pkl`.

- **Web Application**:  
  After running the Streamlit app (`stream.py`), use the sliders to adjust measurements and visualize the predictions. The model will predict whether the tumor is benign or malignant.

## Contribution
Feel free to contribute to the project by creating pull requests or raising issues.

## License
Specify your project license (e.g., MIT License).


