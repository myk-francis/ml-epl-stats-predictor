from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import pickle, datetime
import pandas as pd

import models, ml_utils, database

models.Base.metadata.create_all(bind=database.engine)
app = FastAPI()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. Initial training
@app.post("/train-initial/")
def train_initial(csv_path: str, db: Session = Depends(get_db)):
    df = pd.read_csv(csv_path)
    model, encoder, acc = ml_utils.train_model(df)

    # Save data
    for _, row in df.iterrows():
        match = models.MatchData(**row.to_dict())
        db.add(match)
    db.commit()

    # Save model
    model_bytes = pickle.dumps(model)
    encoder_bytes = pickle.dumps(encoder)
    m = models.ModelStore(model=model_bytes, encoders=encoder_bytes, accuracy=acc)
    db.add(m)
    db.add(models.Logs(message=f"Initial training done. Accuracy: {acc:.2f}"))
    db.commit()
    return {"accuracy": acc}

# 2. Predict
@app.post("/predict/")
def predict_games_api(games: list[tuple], db: Session = Depends(get_db)):
    latest_model = db.query(models.ModelStore).order_by(models.ModelStore.trained_at.desc()).first()
    model = pickle.loads(latest_model.model)
    encoder = pickle.loads(latest_model.encoders)

    today = datetime.date.today()
    weekday = today.weekday()  # Monday=0
    month = today.month
    year = today.year

    df_preds = ml_utils.predict_games(model, encoder, games, year, month, weekday)
    return df_preds.to_dict(orient="records")

# 3. Retrain with new data
@app.post("/train-new/")
def train_new(records: list[dict], db: Session = Depends(get_db)):
    # Insert new data
    for rec in records:
        db.add(models.MatchData(**rec))
    db.commit()

    # Retrain on full dataset
    all_data = pd.read_sql(db.query(models.MatchData).statement, db.get_bind())
    model, encoder, acc = ml_utils.train_model(all_data)

    m = models.ModelStore(model=pickle.dumps(model), encoders=pickle.dumps(encoder), accuracy=acc)
    db.add(m)
    db.add(models.Logs(message=f"Retrained model. Accuracy: {acc:.2f}"))
    db.commit()
    return {"accuracy": acc}

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
    logs = db.query(models.Logs).filter(models.Logs.created_at == datetime.datetime.strptime(date, "%Y-%m-%d").date()).all()
    return [{"time": l.created_at, "msg": l.message} for l in logs]
