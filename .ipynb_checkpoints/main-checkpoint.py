from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
import time
from enum import Enum


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Parcelle(str, Enum):
    Amandier_35 = "Amandier 35"
    Apiculture = "Apiculture"
    Arbequina_10 = "Arbequina 10"
    Arbequina_11_5 = "Arbequina 11,5"
    Arbequina_5_5 = "Arbequina 5,5"
    Jardin = "Jardin"
    KARMA_SERVICES = "KARMA SERVICES"
    Koroneiki_oued = "Koroneiki oued"
    Vanne_1 = "Vanne 1"
    Vanne_2 = "Vanne 2"
    Vanne_3 = "Vanne 3"
    Vanne_4_5_6 = "Vanne 4-5-6"
    Vigne_1 = "Vigne 1"
    Vigne_2 = "Vigne 2"


class Culture(str, Enum):
    Apiculture = "Apiculture"
    Jardin = "Jardin"
    Olivier = "Olivier"
    Vigne = "Vigne"


class Model(BaseModel):
    quantite_ch_PPC: float
    cout_unitaire_ch_PPC: float
    ppc_nbr_ha: float
    ppc_nbr_pieds: float
    date_numeric: float

    parcelle_ch_PPC: Parcelle  # 14 variables
    culture_ch_PPC: Culture  # 4 variables


@app.post("/")
async def root(data: Model):
    x = {
        "quantite_ch_PPC": data.quantite_ch_PPC,
        "cout_unitaire_ch_PPC": data.cout_unitaire_ch_PPC,
        "ppc_nbr_ha": data.ppc_nbr_ha,
        "ppc_nbr_pieds": data.ppc_nbr_pieds,
        "date_numeric": data.date_numeric,
        "culture_ch_PPC_Apiculture": data.culture_ch_PPC == Culture.Apiculture,
        "culture_ch_PPC_Jardin": data.culture_ch_PPC == Culture.Jardin,
        "culture_ch_PPC_Olivier": data.culture_ch_PPC == Culture.Olivier,
        "culture_ch_PPC_Vigne": data.culture_ch_PPC == Culture.Vigne,
        "parcelle_ch_PPC_Amandier 35": data.parcelle_ch_PPC == Parcelle.Amandier_35,
        "parcelle_ch_PPC_Apiculture": data.parcelle_ch_PPC == Parcelle.Apiculture,
        "parcelle_ch_PPC_Arbequina 10": data.parcelle_ch_PPC == Parcelle.Arbequina_10,
        "parcelle_ch_PPC_Arbequina 11,5": data.parcelle_ch_PPC
        == Parcelle.Arbequina_11_5,
        "parcelle_ch_PPC_Arbequina 5,5": data.parcelle_ch_PPC == Parcelle.Arbequina_5_5,
        "parcelle_ch_PPC_Jardin": data.parcelle_ch_PPC == Parcelle.Jardin,
        "parcelle_ch_PPC_KARMA SERVICES": data.parcelle_ch_PPC
        == Parcelle.KARMA_SERVICES,
        "parcelle_ch_PPC_Koroneiki oued": data.parcelle_ch_PPC
        == Parcelle.Koroneiki_oued,
        "parcelle_ch_PPC_Vanne 1": data.parcelle_ch_PPC == Parcelle.Vanne_1,
        "parcelle_ch_PPC_Vanne 2": data.parcelle_ch_PPC == Parcelle.Vanne_2,
        "parcelle_ch_PPC_Vanne 3": data.parcelle_ch_PPC == Parcelle.Vanne_3,
        "parcelle_ch_PPC_Vanne 4-5-6": data.parcelle_ch_PPC == Parcelle.Vanne_4_5_6,
        "parcelle_ch_PPC_Vigne 1": data.parcelle_ch_PPC == Parcelle.Vigne_1,
        "parcelle_ch_PPC_Vigne 2": data.parcelle_ch_PPC == Parcelle.Vigne_2,
    }
    x = list(x.values())
    print(x)

    with open("./models/random_forest_model.sav", "rb") as dt:
        dt = float(pickle.load(dt).predict([x]).flatten()[0])
        return {
            "results": round(dt, 2),
        }
