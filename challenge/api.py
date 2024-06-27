from fastapi import FastAPI, HTTPException
import pandas as pd

from challenge.model import DelayModel

from pydantic import BaseModel, validator
from typing import List

app = FastAPI()
delay_model = DelayModel()


class Flight(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str

    @validator('MES')
    def validate_month(cls, v):
        if v < 1 or v > 12:
            raise HTTPException(
                status_code=400,
                detail='MES must be between 1 and 12'
            )
        return v

    @validator('TIPOVUELO')
    def validate_flight_type(cls, v):
        if v not in ['I', 'N']:
            raise HTTPException(
                status_code=400,
                detail='TIPOVUELO must be either "I" or "N"'
            )
        return v


class PredictionInfo(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", response_model=dict, status_code=200)
async def post_predict(input: PredictionInfo) -> dict:
    try:
        data = [
            {
                "OPERA": flight.OPERA,
                "MES": flight.MES,
                "TIPOVUELO": flight.TIPOVUELO
            } for flight in input.flights
        ]
        df = pd.DataFrame(data)

        preprocessed_data = delay_model.preprocess(df)
        predictions = delay_model.predict(preprocessed_data)

        return {"predict": predictions}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the prediction: {str(e)}"
        )
