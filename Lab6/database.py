from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_test.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StressLog(Base):
    __tablename__ = "stress_predictions"
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String)
    sleep_hours = Column(Float)
    social_media_hours = Column(Float)
    predicted_stress = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()