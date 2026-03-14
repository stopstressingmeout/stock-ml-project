import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1 Load data
df = pd.read_csv("stock_data.csv")

print(df.head())
print("Shape:", df.shape)

# 2 Preprocessing
df['Date'] = pd.to_datetime(df['Unnamed: 0'])
df.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values
df.ffill(inplace=True)

# Feature engineering
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)

# Features and target
X = df.drop(columns=['Stock_5'])
y = df['Stock_5']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# 4 Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50,100,200],
    'model__max_depth': [None,5,10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)

# 5 Train
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# 6 Cross validation
scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5)

print("CV mean:", scores.mean())
print("CV std:", scores.std())

# 7 Test evaluation
pred = grid.best_estimator_.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)

# 8 Save model
joblib.dump(grid.best_estimator_, "model.pkl")

print("Model saved as model.pkl")