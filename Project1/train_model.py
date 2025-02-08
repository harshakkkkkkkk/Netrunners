import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate Synthetic Data
num_samples = 1000
data = {
    "voltage": np.random.uniform(210, 250, num_samples),
    "current": np.random.uniform(0.5, 15, num_samples),
    "temperature": np.random.uniform(20, 80, num_samples),
    "humidity": np.random.uniform(30, 80, num_samples),
    "fault": [random.choice([0, 1]) for _ in range(num_samples)]  # 1 = Fault, 0 = No Fault
}

df = pd.DataFrame(data)

# Split Data
X = df.drop(columns=["fault"])
y = df["fault"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save Model
with open("fault_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete! Saved as fault_model.pkl.")