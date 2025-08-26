import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Path to data file
DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA", "epl_preprocessed.csv")

def load_initial_data():
    """Loads the CSV from the data folder"""
    df = pd.read_csv(DATA_PATH)
    return df

def train_model(df: pd.DataFrame, target_col: str, threshold: int):
    # Binary classification target
    df["Target"] = (df[target_col] >= threshold).astype(int)

    X = df[["HomeTeam", "AwayTeam", "Year", "Month", "Weekday"]]
    y = df["Target"]

    # Encode categorical features
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



def predict_games_multi(task_models, encoder, games, year, month, weekday):
    df = pd.DataFrame(games, columns=["HomeTeam", "AwayTeam"])
    df["Year"] = year
    df["Month"] = month
    df["Weekday"] = weekday

    X_enc = encoder.transform(df[["HomeTeam", "AwayTeam", "Year", "Month", "Weekday"]])

    results = []
    for i, row in df.iterrows():
        game_result = {
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"]
        }
        for task_name, model in task_models.items():
            pred = model.predict(X_enc[i])
            if task_name == "goals":
                game_result["GoalsPrediction"] = "2 or more goals" if pred[0] == 1 else "Under 2 goals"
            elif task_name == "bookings":
                game_result["BookingsPrediction"] = "3 or more bookings" if pred[0] == 1 else "Under 3 bookings"
            elif task_name == "corners":
                game_result["CornersPrediction"] = "10 or more corners" if pred[0] == 1 else "Under 10 corners"
        results.append(game_result)

    return pd.DataFrame(results)

