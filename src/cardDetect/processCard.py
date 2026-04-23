from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

router = APIRouter()
model = keras.model.load_model("./fakeCardCNN.keras") #figure out where I put .keras file
classNames = [
    "real",
    "fake"
]

@app.post("/predict", summary = "Predict Card Class")
# I will need to figure out if it is even a valid card to be processed in the first place. Right now, that functionality isn't really supported.
async def predictCard(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", "jpeg")):
        raise HTTPException(
            status_code = 400,
            detail = "Invalid image format. Only PNG and JPG are allowed.",
        )
    try:
        rawImage = await file.read()
        image = Image.open(io.BytesIO(rawImage).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing images.")
    
    #preprocess the image 
    inputToTensor = tf.cast(tf.image.resize(image, (224, 224)), tf.float32) / 255.0
    return JSONResponse({"predicted_class": classNames[model.predict(inputToTensor)]})
    