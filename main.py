from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import seaborn as sns
from fastapi.responses import StreamingResponse
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

from keras.models import model_from_json

import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    # print(len(dataY))
    return np.array(dataX), np.array(dataY)


@app.post("/")
async def root(file: UploadFile):
    file_content = file.file.read()
    decoded_content = file_content.decode("utf-8")
    df2 = pd.read_csv(StringIO(decoded_content))
    cycle = df2["cycle"]
    cycle = cycle[: len(cycle) - 1].tolist()

    Test_dataset = df2["SOH"]

    Test_dataset = np.array(Test_dataset)
    Test_dataset = Test_dataset.reshape((len(Test_dataset), 1))
    look_back = 1

    testX, testY = create_dataset(Test_dataset, look_back)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # print(trainX.shape)
    # print(testX.shape)

    chemin_json = "B05_model.json"
    chemin_weights = "B05_weights.h5"

    # Chargez l'architecture du modèle depuis le fichier JSON
    with open(chemin_json, "r") as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)
    # Chargez les poids du modèle depuis le fichier H5
    loaded_model.load_weights(chemin_weights)

    yhat = loaded_model.predict(testX)[:, 0].tolist()

    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))
    plt.plot(cycle, yhat, label="LSTM Prediction", linewidth=3, color="r")
    plt.legend(prop={"size": 16})
    plt.ylabel("SoH", fontsize=15)
    plt.xlabel("Discharge cycle", fontsize=15)
    plt.title(" SOH Prediction", fontsize=15)
    plt.savefig("imLSTM.jpg")

    image_buffer = BytesIO()
    plt.savefig(image_buffer, format="png")
    image_buffer.seek(0)

    return StreamingResponse(BytesIO(image_buffer.read()), media_type="image/png")
