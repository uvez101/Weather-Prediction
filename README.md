# 🌊 New Mansoura Deep Learning Forecaster

An end-to-end Machine Learning pipeline that predicts the 7-day daily high temperature trend for the Mediterranean coastal city of New Mansoura, Egypt. 

This project evolved from a static Kaggle dataset using standard regression to a fully automated pipeline utilizing live API data and a PyTorch Deep Learning architecture.


## 🔗 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uvez-weather-prediction-deeplearning.streamlit.app/)

Check out the interactive model here: [View App](https://uvez-weather-prediction-deeplearning.
streamlit.app/)

## 🧠 Project Architecture

This application is built on a modern AI/ML tech stack, separated into three distinct pipeline stages: Data Engineering, Model Training, and Web Deployment.

### 1. Data Engineering (ETL)
* **Source:** 10 years of historical weather data fetched directly via the **Open-Meteo REST API**.
* **Processing:** Uses `pandas` to aggregate hourly data into Daily Maximums (predicting the peak heat of the day, rather than a mathematical average).
* **Feature Engineering:** Constructs a 7-day sliding window target and introduces a `month` feature to capture complex Mediterranean seasonality. Includes features for Temperature, Humidity, Wind Speed, and Precipitation.

### 2. Deep Learning Core (PyTorch)
* **Algorithm:** Long Short-Term Memory (LSTM) Neural Network.
* **Why LSTM?:** Upgraded from traditional tree-based models (Random Forest/XGBoost) to capture the sequential, time-series nature of coastal weather patterns, reducing the Mean Absolute Error (MAE) significantly.
* **Scaling:** Utilizes Scikit-Learn's `MinMaxScaler` to compress data into 3D tensors `[Samples, Time_Steps, Features]` for optimal neural network backpropagation.
* *Note: The model was trained using an Nvidia T4 GPU via Google Colab to handle the 1500-epoch training loop.*

### 3. Deployment & UI
* **Frontend:** Built with **Streamlit** for a clean, responsive, and interactive user interface.
* **Hosting:** Deployed live via Streamlit Cloud using a CI/CD pipeline linked to this GitHub repository.
* **Environment:** Enforces strict dependency management (`requirements.txt`) to ensure PyTorch and Streamlit compatibility on cloud CPU servers.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Deep Learning:** PyTorch (`torch`, `torch.nn`)
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (`MinMaxScaler`)
* **Frontend/Hosting:** Streamlit

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/uvez101/Weather-Prediction.git](https://github.com/uvez101/Weather-Prediction.git)
   cd Weather-Prediction