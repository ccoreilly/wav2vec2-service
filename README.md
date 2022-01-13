# Wav2Vec2 simple service

Mimics the Huggingface Inference API for a speech recognition model.

## Usage

This is an example with an image which includes a Wav2Vec2 model for the catalan language

```sh
> docker run -p 8000:8000 -d ghcr.io/ccoreilly/wav2vec2-catala:0.1.0
> curl -X POST localhost:8000/recognize -F "file=@sample.wav"
{"text":"bon vesprà a totes i tots donem començament al ple ordinari convocat per avui trenta de setembre de dos mil vint-i-u a les vuit hores en el saló de plens d'ací de l'ajuntament de massanassa"}
```

## Converting Wav2Vec2 to ONNX format

Using the ONNX model format results in an increase in inference speed when using a CPU. You can convert any Wav2Vec2ForCTC model from the huggingface model hub using the `convert_torch_to_onnx.py` script:

```sh
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> python convert_torch_to_onnx.py --model ccoreilly/wav2vec2-large-xlsr-catala
```

You can also quantize the model to reduce its size

```sh
> python convert_torch_to_onnx.py --model ccoreilly/wav2vec2-large-xlsr-catala --quantize
```
