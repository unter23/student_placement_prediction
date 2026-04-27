import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/placement.csv")

# Convert Yes/No to 1/0
df['Placed'] = df['Placed'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop("Placed", axis=1)
y = df["Placed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
