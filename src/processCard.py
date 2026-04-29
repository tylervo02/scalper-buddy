from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

router = APIRouter()
model = keras.models.load_model("./fakeCardCNN.keras") #figure out where I put .keras file
classNames = [
    "real",
    "fake"
]

@router.get("/")
def root():
    return {"status": "API is running"}


@router.post("/predict", summary = "Predict Card Class")
# I will need to figure out if it is even a valid card to be processed in the first place. Right now, that functionality isn't really supported.
async def predictCard(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code = 400,
            detail = "Invalid image format. Only PNG and JPG are allowed.",
        )
    try:
        rawImage = await file.read()
        image = Image.open(io.BytesIO(rawImage)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing images.")
    
    #preprocess the image 
    inputToTensor = tf.convert_to_tensor(image)
    inputToTensor = tf.image.resize(inputToTensor, (224, 224))
    inputToTensor = tf.cast(inputToTensor, tf.float32) / 255.0
    inputToTensor = tf.expand_dims(inputToTensor, axis=0)

    # return JSONResponse({"predicted_class": classNames[int(tf.argmax(model.predict(inputToTensor)[0]))]})

    pred = model.predict(inputToTensor)[0]
    return JSONResponse({
        "fakechance": float(pred[0])
    })