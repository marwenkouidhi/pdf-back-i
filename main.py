import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Model(BaseModel):
    quantite_ch_PPC: float
    cout_unitaire_ch_PPC: float
    mois: int
    jour: int
    activite_ch_PPC: str


@app.post("/")
async def root(data: Model):
    sample = pd.DataFrame(
        [dict(data)],
    )
    encoder = joblib.load("./models/encoder_model.sav")

    encoded_sample = encoder.transform(sample[["activite_ch_PPC", "mois", "jour"]])

    # Create a DataFrame with the encoded features
    encoded_sample = pd.DataFrame(
        encoded_sample,
        columns=encoder.get_feature_names_out(["activite_ch_PPC", "mois", "jour"]),
    )

    # Concatenate the encoded features DataFrame with the original DataFrame
    encoded_sample = pd.concat([sample, encoded_sample], axis=1)
    encoded_sample.drop(columns=["activite_ch_PPC", "mois", "jour"], inplace=True)

    model = joblib.load("models/model.sav")
    res = float(model.predict(encoded_sample).flatten()[0])

    return round(res, 2)
