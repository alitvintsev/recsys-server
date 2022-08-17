from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from app.utils import ModelInference


# @dataclass
# class Prediction(BaseModel):
#     restaurant_name: str
#     avg_rating: float


app = FastAPI()

model = ModelInference()


@app.get("/")
async def read_root():
    return {"Test": "Working"}


@app.get("/predict/{text}")
async def get_prediction(text):
    prediction = model.get_prediction(text)
    return prediction



# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

    # prediction = model.get_prediction(text)
    # return prediction