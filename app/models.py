from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.sql import func
from database import Base


class MatchData(Base):
    __tablename__ = "match_data"
    id = Column(Integer, primary_key=True, index=True)
    HomeTeam = Column(String)
    AwayTeam = Column(String)
    Year = Column(Integer)
    Month = Column(Integer)
    Weekday = Column(Integer)
    TotalGoals = Column(Integer)
    TotalBookings = Column(Integer)
    TotalCorners = Column(Integer)

class ModelStore(Base):
    __tablename__ = "model_store"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)   # "goals", "bookings", "corners"
    model = Column(LargeBinary)
    encoders = Column(LargeBinary)
    accuracy = Column(Float)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())

class Logs(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
