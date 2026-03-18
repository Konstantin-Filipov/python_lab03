#Author: Konstantin Filipov
#Date: 2026-02-28
#Task A.3.1: Handwriting Recognition

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# loading training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
#loading test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

#---------------------------------------------------------------------------------------------------------------------------
# Task I: Linear Reg, SVM & RF (10 output classes)--------------------------------------------------------------------------

#convert data to numpy & reshape
X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

subset = 10000  # get a subset of the data for faster training
X_train_small = X_train[:subset]
y_train_small = y_train[:subset]

# Linear reg
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_small, y_train_small)
y_pred_lr = lin_reg_model.predict(X_test)
#Convert regression outputs to nearest digit
y_pred_lr = np.round(y_pred_lr).astype(int)
y_pred_lr = np.clip(y_pred_lr, 0, 9)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

# SVM kernel linear model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_small, y_train_small)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)

#Random Forest
rf_model = RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=-1)
rf_model.fit(X_train_small, y_train_small)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nModel Performance")
print("-----------------")
print(f"Linear Regression Accuracy: {accuracy_lr:.4f}")
print(f"SVM (Linear) Accuracy: {accuracy_svm:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

#---------------------------------------------------------------------------------------------------------------------------
# Task II: Visualize MSE error per epoch -----------------------------------------------------------------------------------

epochs = 10
step = subset // epochs # number of samples per epoch
mse_lr, mse_svm, mse_rf = [], [], [] # lists to store MSE values

# Train models incrementally and calculate MSE on test set after each epoch
for epoch in range(1, epochs + 1):
    size = epoch * step
    size = max(size, 1)# ensure at least 1 sample is used in the first epoch
    X_epoch = X_train_small[:size]
    y_epoch = y_train_small[:size]

    #Linear regression
    lin_reg_model.fit(X_epoch, y_epoch)
    pred_lr = lin_reg_model.predict(X_test)
    pred_lr = np.round(pred_lr).astype(int)
    pred_lr = np.clip(pred_lr, 0, 9)
    mse_lr.append(mean_squared_error(y_test, pred_lr))

    #SVM
    svm_model.fit(X_epoch, y_epoch)
    pred_svm = svm_model.predict(X_test)
    mse_svm.append(mean_squared_error(y_test, pred_svm))

    #Random Forest
    rf_model.fit(X_epoch, y_epoch)
    pred_rf = rf_model.predict(X_test)
    mse_rf.append(mean_squared_error(y_test, pred_rf))

#visualize MSE vs Epoch for all models
plt.figure()

plt.plot(range(1, epochs + 1), mse_lr, label="LR")
plt.plot(range(1, epochs + 1), mse_svm, label="SVM")
plt.plot(range(1, epochs + 1), mse_rf, label="RF")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch for SVM, LR, RF")

plt.legend(loc="upper right")

plt.show()