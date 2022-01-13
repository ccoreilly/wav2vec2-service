from os import getenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from transformers import Wav2Vec2Processor
import onnxruntime as rt
import soundfile as sf
import numpy as np

processor = Wav2Vec2Processor.from_pretrained("ccoreilly/wav2vec2-large-100k-voxpopuli-catala")

ONNX_PATH = getenv("model_path", "wav2vec2.onnx")
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
session = rt.InferenceSession(ONNX_PATH, sess_options)

def predict(file):
  speech_array, sr = sf.read(file)
  features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
  input_values = features.input_values
  onnx_outputs = session.run(None, {session.get_inputs()[0].name: input_values.numpy()})[0]
  prediction = np.argmax(onnx_outputs, axis=-1)
  return processor.decode(prediction.squeeze().tolist())

app = FastAPI()

@app.get('/health')
async def health_check():
    return 'OK'

@app.post('/recognize')
async def recognize(file: UploadFile = File(..., media_type="audio/wav")):
    if file:
      return {
        "text": predict(file.file)
      }
    else:
        raise HTTPException(status_code=400, detail="Audio bytes expected")


