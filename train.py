# titanic_train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the Titanic dataset
data = pd.read_csv('titanic.csv')

# Preview data
print(data.head())

# Data Preprocessing
# 1. Fill missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# 2. Convert categorical variables to numeric
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# 3. Drop unnecessary columns
data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# 4. Define features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# 5. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'titanic_model.pkl')

