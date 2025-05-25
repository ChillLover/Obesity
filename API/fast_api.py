from fastapi import FastAPI
import joblib
import json
from sklearn.multiclass import OneVsRestClassifier
from catboost import CatBoostClassifier
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Optional

app = FastAPI()


class DataToPredictStr(BaseModel):
    data: str


class Row(BaseModel):
    Gender: Optional[str] = Field(None, description="Gender of a person")
    Age: float = Field(..., description="Age of a person")
    Height: float = Field(..., description="Height of a person in centimeters")
    Weight: float = Field(..., description="Weight of a person in kilograms")
    family_history: str = Field(..., description="Family history of overweight (yes/no)")
    FAVC: str = Field(..., description="Frequent consumption of high caloric food (yes/no)")
    FCVC: float = Field(..., description="Frequency of consumption of vegetables (scale 1-3)")
    NCP: float = Field(..., description="Number of main meals per day")
    CAEC: str = Field(..., description="Consumption of food between meals (no/Sometimes/Frequently/Always)")
    SMOKE: str = Field(..., description="Smoking habit (yes/no)")
    CH2O: float = Field(..., description="Daily water consumption (scale 1-3)")
    SCC: str = Field(..., description="Calories consumption monitoring (yes/no)")
    FAF: float = Field(..., description="Physical activity frequency (scale 1-3)")
    TUE: float = Field(..., description="Time using technology devices (scale 1-3)")
    CALC: str = Field(..., description="Alcohol consumption (no/Sometimes/Frequently/Always)")
    MTRANS: str = Field(..., description="Transportation used (Public/Bike/Walking/Motorbike/Automobile)")


class DataToPredictControled(BaseModel):
    data: List[Row] = Field(..., description="All features together")


model = joblib.load("model.pkl")


# @app.post("/predict_input_features")
# async def predict_imput_features(raw_data: Row):
#     df = pd.DataFrame.from_dict(json.loads(raw_data.model_dump_json())["data"])
#     preds = pd.Series(model.predict(df))

#     return {"Answer": preds.to_json(orien="records")}


@app.post("/predict_features")
async def predict_features(raw_data: DataToPredictControled):
    df = pd.DataFrame.from_dict(json.loads(raw_data.model_dump_json())["data"])
    preds = pd.Series(model.predict(df))

    return {"Answer": preds.to_json(orient="records")}


@app.post("/predict")
async def predict(raw_data: DataToPredictStr):
    df = pd.read_json(raw_data.data, orient="records")
    preds = pd.Series(model.predict(df))

    return {"Answer": preds.to_json(orient="records")}