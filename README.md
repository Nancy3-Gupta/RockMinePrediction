# Rock vs Mine Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv(r'C:\\Users\\gupta\\OneDrive\\Desktop\\Tensorflow\\Copy of sonar data (1).csv')

# Separate features and labels
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify=y, random_state=2, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)  # Apply same transformation

# Train Logistic Regression model
model = LogisticRegression(max_iter=5000)  # Increased max_iter to avoid warnings
model.fit(xtrain, ytrain)

# Evaluate model
train_pred = model.predict(xtrain)
test_pred = model.predict(xtest)
train_accuracy = accuracy_score(ytrain, train_pred)
test_accuracy = accuracy_score(ytest, test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Making Predictions
input_data = (0.0131, 0.0201, 0.0045, 0.0217, 0.023, 0.0481, 0.0742, 0.0333, 0.1369, 0.2079, 
              0.2295, 0.199, 0.1184, 0.1891, 0.2949, 0.5343, 0.685, 0.7923, 0.822, 0.729, 
              0.7352, 0.7918, 0.8057, 0.4898, 0.1934, 0.2924, 0.6255, 0.8546, 0.8966, 0.7821, 
              0.5168, 0.484, 0.4038, 0.3411, 0.2849, 0.2353, 0.2699, 0.4442, 0.4323, 0.3314, 
              0.1195, 0.1669, 0.3702, 0.3072, 0.0945, 0.1545, 0.1394, 0.0772, 0.0615, 0.023, 
              0.0111, 0.0168, 0.0086, 0.0045, 0.0062, 0.0065, 0.003, 0.0066, 0.0029, 0.0053)

# Convert input data to numpy array and reshape
input_array = np.asarray(input_data).reshape(1, -1)

# Apply the same scaling as training
input_array = scaler.transform(input_array)

# Predict
prediction = model.predict(input_array)[0]

print("Prediction:", prediction)
