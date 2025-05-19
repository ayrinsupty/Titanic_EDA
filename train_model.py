import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/train.csv')

# Data cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True, errors='ignore')

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'AgeBin', 'FareBin'], drop_first=True)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Scale numerical columns
scaler = StandardScaler()
num_cols = [col for col in ['Age', 'Fare', 'FamilySize'] if col in X.columns]
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Training complete and files saved: model.pkl, scaler.pkl")
