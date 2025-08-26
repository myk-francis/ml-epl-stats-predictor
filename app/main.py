from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func  # <-- import func here
import pickle, datetime
import pandas as pd

import models, ml_utils, database

models.Base.metadata.create_all(bind=database.engine)
app = FastAPI()

# allow your frontend domain here
origins = [
    "https://drinking-games-hub.vercel.app",  # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. Initial training
@app.post("/train-initial/")
def train_initial(db: Session = Depends(get_db)):
    # Load CSV data from data/ folder
    df = ml_utils.load_initial_data()

    # Keep columns to save into DB
    df_to_save = df[["HomeTeam", "AwayTeam", "Year", "Month", "Weekday", 
                     "TotalGoals", "TotalBookings", "TotalCorners"]]

    # Save match data into DB
    for _, row in df_to_save.iterrows():
        match = models.MatchData(**row.to_dict())
        db.add(match)
    db.commit()

    # Define tasks -> (column, threshold)
    tasks = {
        "goals": ("TotalGoals", 2),       # >= 2 goals
        "bookings": ("TotalBookings", 3), # >= 3 bookings
        "corners": ("TotalCorners", 10)   # >= 10 corners
    }

    results = {}

    for task_name, (col, threshold) in tasks.items():
        model, encoder, acc = ml_utils.train_model(df, col, threshold)

        # Save each trained model
        m = models.ModelStore(
            model_name=task_name,
            model=pickle.dumps(model),
            encoders=pickle.dumps(encoder),
            accuracy=acc
        )
        db.add(m)
        db.add(models.Logs(message=f"Initial training for {task_name} done. Accuracy: {acc:.2f}"))
        results[task_name] = {"accuracy": acc}

    db.commit()
    return results




# 2. Predict
@app.post("/predict/")
def predict_games_api(games: list[tuple], db: Session = Depends(get_db)):
    # Load all models
    models_list = db.query(models.ModelStore).order_by(models.ModelStore.trained_at.desc()).all()

    # Pick the latest version of each model type
    task_models = {}
    encoder = None
    for m in models_list:
        if m.model_name not in task_models:  # first one encountered is latest (due to desc order)
            task_models[m.model_name] = pickle.loads(m.model) # type: ignore
            if encoder is None:  # all models share the same encoder
                encoder = pickle.loads(m.encoders) # type: ignore

    today = datetime.date.today()
    weekday = today.weekday()  # Monday=0
    month = today.month
    year = today.year

    # Run predictions for each model
    df_preds = ml_utils.predict_games_multi(task_models, encoder, games, year, month, weekday)

    return df_preds.to_dict(orient="records")


# 3. Retrain with new data
@app.post("/train/")
def train_incremental(new_data: list[dict], db: Session = Depends(get_db)):
    """
    Add new match data with a Date field, save to DB, and retrain all models.
    """
    processed_rows = []
    for row in new_data:
        # Parse date
        date_obj = datetime.datetime.strptime(row["Date"], "%Y-%m-%d")
        row["Year"] = date_obj.year
        row["Month"] = date_obj.month
        row["Weekday"] = date_obj.weekday()  # Monday=0, Sunday=6
        del row["Date"]  # no need to store raw Date since we break it down
        processed_rows.append(row)

    # 1. Save new data into DB
    for row in processed_rows:
        match = models.MatchData(**row)
        db.add(match)
    db.commit()

    if db.bind is None:
        raise ValueError("Database bind is not available.")

    # 2. Load all data back from DB
    all_data = pd.read_sql(db.query(models.MatchData).statement, db.bind)

    # 3. Train models for goals, bookings, corners
    tasks = {
        "goals": ("TotalGoals", 2),       # >= 2 goals
        "bookings": ("TotalBookings", 3), # >= 3 bookings
        "corners": ("TotalCorners", 10)   # >= 10 corners
    }

    results = {}
    for task_name, (col, threshold) in tasks.items():
        model, encoder, acc = ml_utils.train_model(all_data, col, threshold)

        # Save model in DB
        m = models.ModelStore(
            model_name=task_name,
            model=pickle.dumps(model),
            encoders=pickle.dumps(encoder),
            accuracy=acc
        )
        db.add(m)
        db.add(models.Logs(message=f"Retrained {task_name} model. Accuracy: {acc:.2f}"))
        results[task_name] = acc

    db.commit()
    return {"accuracies": results}


# 4. Status
@app.get("/status/")
def status(db: Session = Depends(get_db)):
    latest_model = db.query(models.ModelStore).order_by(models.ModelStore.trained_at.desc()).first()
    count = db.query(models.MatchData).count()
    return {
        "last_trained": latest_model.trained_at if latest_model else None,
        "records": count,
        "accuracy": latest_model.accuracy if latest_model else None,
    }

# 5. Logs
@app.get("/logs/")
def get_logs(date: str, db: Session = Depends(get_db)):
    # Parse the date string into a datetime.date object
    target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    logs = db.query(models.Logs).filter(func.date(models.Logs.created_at) == target_date).all()
    return [{"time": l.created_at, "msg": l.message} for l in logs]
