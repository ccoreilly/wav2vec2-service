from os import getenv
from fastapi import FastAPI, HTTPException, File
from transformers import AutoTokenizer, AutoFeatureExtractor, AutomaticSpeechRecognitionPipeline, Wav2Vec2ForCTC

model_name = getenv("model")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
pipe = AutomaticSpeechRecognitionPipeline(feature_extractor=feature_extractor, model=model, tokenizer=tokenizer)

app = FastAPI()

@app.get('/health')
async def health_check():
    return 'OK'

@app.post('/recognize')
async def recognize(file: bytes = File(..., media_type="audio/wav")):
    if file:
        return pipe(file)
    else:
        raise HTTPException(status_code=400, detail="Audio bytes expected")