import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv("data/train.csv")

# Basic cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Scale
scaler = StandardScaler()
X_scaled = X.copy()
num_cols = ['Age', 'Fare', 'FamilySize']
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

# Train Model
model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
