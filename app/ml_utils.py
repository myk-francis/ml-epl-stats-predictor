import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Path to data file
DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA", "epl_goals_2.csv")

def load_initial_data():
    """Loads the CSV from the data folder"""
    df = pd.read_csv(DATA_PATH)
    return df

def train_model(df: pd.DataFrame):
    df["Target"] = (df["TotalGoals"] >= 2).astype(int)
    X = df[["HomeTeam", "AwayTeam", "Month", "Weekday"]]
    y = df["Target"]

    # Encode
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_enc = encoder.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, encoder, acc

def predict_games(model, encoder, games, year, month, weekday):
    df = pd.DataFrame(games, columns=["HomeTeam", "AwayTeam"])
    df["Year"] = year
    df["Month"] = month
    df["Weekday"] = weekday
    X_enc = encoder.transform(df[["HomeTeam", "AwayTeam", "Month", "Weekday"]])
    preds = model.predict(X_enc)

    df["Prediction"] = ["2 or more goals" if p == 1 else "Under 2 goals" for p in preds]
    return df
