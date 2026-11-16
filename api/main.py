

import socket
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

VERSION = "2.1"
HOTE = socket.gethostname()

# Chargement du mod??le
modele = joblib.load("models/modele.pkl")

app = FastAPI(title="API de detection de fraude")


class Transaction(BaseModel):
    montant: float
    anciennete: int
    freq_24h: int
    nuit: int


@app.get("/health")
def health():
    return {
        "statut": "ok",
        "version": VERSION,
        "hote": HOTE
    }


@app.post("/predict")
def predict(t: Transaction):
    X = pd.DataFrame([t.model_dump()])
    p = float(modele.predict_proba(X)[0, 1])

    return {
        "fraude": p > 0.5,
        "probabilite": round(p, 4),
        "version": VERSION,
        "hote": HOTE
    }

# modification
# modification 2
