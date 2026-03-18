# Weather Prediction & Clustering Lab  
Machine Learning Lab Assignment – PyTorch & Scikit-Learn

This project contains multiple machine learning tasks implemented using **PyTorch** and **Scikit-Learn**

## 📦 Requirements

Make sure the following are installed:

- Python 3.9+
- PyTorch
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

---

## ⚙️ Setup (Virtual Environment)

This project uses a **Python virtual environment (venv)**.

### 1️⃣ Create virtual environment

```bash
python -m venv venv
```

### 2️⃣ Activate virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib
```

---

### 4️⃣ Select Python Interpreter (IMPORTANT)

If using **VS Code**:

1. Press `Ctrl + Shift + P`
2. Select:

```
Python: Select Interpreter
```

3. Choose:

```
venv/Scripts/python.exe
```

If interpreter is not selected correctly → PyTorch imports may fail.

---

## 📊 Datasets

The following datasets are used:

- **Seattle Weather Dataset** → Supervised Learning
- **Penguins Dataset** → Unsupervised Clustering

Place datasets in the project root directory.

---

## 🚀 Running the Project

Example:

```bash
python 02-predict_rain.py
```

or

```bash
python kmeans_penguins.py
```

---

## 🧠 Implemented Models

### Supervised Learning

- PyTorch Linear Regression (custom training loop)
- Support Vector Machine (Scikit-Learn)
- Random Forest (Scikit-Learn)

### Unsupervised Learning

- K-Means clustering
- Cluster vs real species comparison
- 2D visualization of clusters

---

## 📈 Evaluation Methods

- Accuracy score (classification models)
- Mean Squared Error (MSE) tracking per epoch
- Cluster distribution comparison table

---

## ⚠️ Notes

- PyTorch models require **tensor inputs**
- Scikit-Learn models require **NumPy arrays**
- Therefore conversions between tensor ↔ numpy are performed
- Linear Regression is implemented as regression + rounding for classification

---

## 👨‍💻 Author

Konstantin Filipov  
MSc Computer Science (IoT)