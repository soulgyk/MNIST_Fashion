import os
import pickle
import faulthandler
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import uvicorn
from fastapi.responses import HTMLResponse  

faulthandler.enable()

app = FastAPI()
app = FastAPI(
    title="MNIST Fashion Prediction",
    description="Welcome to MNIST Fashion Prediction! This API allows you to predict the type of clothing in an image.",
    version="1.0.0",
)

# Load the trained model
pkl_file_path = os.path.join('/Users/soulgy/Documents/MNIST_Model_Prediction', 'MNISTmodel.pkl')
with open(pkl_file_path, 'rb') as f:
    model = pickle.load(f)

# Define an Enum for class labels and descriptions
class ClothingClass(str, Enum):
    Dress = "Dress"
    Trousers = "Trousers"
    Tshirt = "T-shirt/top"
    Pullover = "Pullover"
    Coat = "Coat"
    Sandal = "Sandal"
    Shirt = "Shirt"
    Sneaker = "Sneaker"
    Bag = "Bag"
    Ankle_boot = "Ankle boot"

# Define the request body model
class Image(BaseModel):
    pixels: list[float]

# Define the response model to include only class_label
class PredictionResponse(BaseModel):
    class_label: ClothingClass

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Welcome to MNIST Fashion Prediction</title>
        </head>
        <body>
            <h1>Welcome to MNIST Fashion Prediction</h1>
            <p>This API allows you to predict the type of clothing in an image.</p>
            <p>Go to <a href="/docs">Swagger Documentation</a></p>
        </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
def predict(image: Image):
    # Convert the input to numpy array
    image_array = np.array(image.pixels).reshape(1, 32, 32, 3)
    # Perform prediction
    prediction = model.predict(image_array)
    # Get the predicted class
    predicted_class = int(np.argmax(prediction))
    # Map the predicted class to its label description
    class_label = ClothingClass(predicted_class)
    return {"class_label": class_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
