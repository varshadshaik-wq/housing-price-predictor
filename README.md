# 🏠 House Price Prediction using Machine Learning

A simple **Machine Learning project** that predicts house prices based on features like number of rooms, size, and location score using **Linear Regression**.

---

## 🚀 Features

* 📊 Generates a sample dataset
* 🧠 Trains a Linear Regression model
* 📉 Evaluates model performance (MAE & R² Score)
* 🔮 Predicts price for a new house
* 📈 Visualizes Actual vs Predicted prices

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```
house-price-prediction-ml/
│── app.py
│── house_data.csv
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/house-price-prediction-ml.git
```

### 2. Navigate to Project Folder

```
cd house-price-prediction-ml
```

### 3. Install Dependencies

```
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ How to Run

```
python app.py
```

---

## 💻 How It Works

The model uses the formula:

Price = (Rooms × 50000) + (Size × 150) + (Location Score × 30000) + Noise

* Data is split into training and testing sets
* Model learns relationships between features and price
* Predictions are made on unseen data

---

## 📊 Model Evaluation

* **Mean Absolute Error (MAE)** → Measures prediction error
* **R² Score** → Measures model accuracy

---

## 🔮 Example Prediction

```
Number of rooms: 3
Size (sq ft): 1500
Location score (1-10): 7
Estimated price: $XXX,XXX
```

---

## 📈 Output Visualization

* Scatter plot of **Actual vs Predicted Prices**
* Helps understand model performance

---

## 📌 Future Improvements

* Use real-world dataset
* Add multiple ML models (Random Forest, XGBoost)
* Build a web app using Flask
* Add model saving/loading



