import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv('seattle-weather.csv')

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Task I: Linear Reg, SVM & RF (5 output classes)-----------------------------------------------------------------------------------------------------

#features & target
X = df[['precipitation','temp_max','temp_min','wind']]#ignore the date column and use the 4 weather features as input
y = df['weather'].astype('category').cat.codes # convert weather categories to numerical codes (0, 1, 2) for clear regression targets
#convert to numpy
X = X.to_numpy()
y = y.to_numpy()
#convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
#split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Linear regression model
lr_model = nn.Sequential(nn.Linear(4, 1))

n_epochs = 100
batch_size = 8
mse_lr = []

# define loss function
loss_fn = nn.MSELoss()

# define optimizer with a spicific learning rate
optimizer = optim.Adam(lr_model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        # take a batch
        Xbatch = X_train[i:i+batch_size]
        ybatch = y_train[i:i+batch_size]
        # forward pass
        y_pred =lr_model(Xbatch) # torch.max(model(Xbatch), 1)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    
    lr_model.eval()
    y_pred = lr_model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = mse.detach().item()
    mse_lr.append(mse)
    
#linear regression accuracy (after rounding to nearest category)
predicted = torch.round(y_pred)
predicted = torch.clamp(predicted, 0, 4) #clip to avoid out-of-range values
lr_acc = (predicted == y_test).float().mean()

#convert tensors back to numpy for SVM
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().ravel()
X_test_np = X_test.numpy()
y_test_np = y_test.numpy().ravel()

#SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train_np, y_train_np)
pred_svm = svm_model.predict(X_test_np)
svm_acc = accuracy_score(y_test_np, pred_svm)

#Random Forest model
rf_model = RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=-1)
rf_model.fit(X_train_np, y_train_np)
y_pred_rf = rf_model.predict(X_test_np)
rf_acc = accuracy_score(y_test_np, y_pred_rf)

print("\nModel Performance")
print("-----------------")
print(f"Linear Regression Accuracy: {lr_acc.item():.4f}")
print(f"SVM (Linear) Accuracy: {svm_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Task II: Visualize MSE error per epoch ---------------------------------------------------------------------------------------------------------------

step = len(X_train_np) // n_epochs

mse_svm, mse_rf = [], []

for epoch in range(n_epochs):
    size = (step + 1) * step
    size = max(size, 1) # ensure at least 1 sample is used in the first epoch

    X_epoch = X_train_np[:size]
    y_epoch = y_train_np[:size]

    #SVM
    svm_model.fit(X_epoch, y_epoch)
    pred_svm = svm_model.predict(X_test_np)
    mse_svm.append(mean_squared_error(y_test_np, pred_svm))

    #Random Forest
    rf_model.fit(X_epoch, y_epoch)
    pred_rf = rf_model.predict(X_test_np)
    mse_rf.append(mean_squared_error(y_test_np, pred_rf))

#visualize MSE vs Epoch for all models
plt.figure()

plt.plot(range(1, n_epochs + 1), mse_lr, label="LR")
plt.plot(range(1, n_epochs + 1), mse_svm, label="SVM")
plt.plot(range(1, n_epochs + 1), mse_rf, label="RF")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch for SVM, LR, RF")

plt.legend(loc="upper right")
plt.grid(True)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Task III: Confusion Matrix for SVC -------------------------------------------------------------------------------------------------------------------
cm=confusion_matrix(y_test_np, pred_svm) # use the trained model to make predictions on the test set.
#Plotting the Confusion Matrix with Labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['weather'].astype('category').cat.categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM")
plt.show()