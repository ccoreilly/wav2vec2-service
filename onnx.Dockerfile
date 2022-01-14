FROM python:3.9.7-slim-buster

ENV model=ccoreilly/wav2vec2-large-100k-voxpopuli-catala

RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg bash curl && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

WORKDIR /app

RUN addgroup --gid 1001 --system wav2vec2 && \
  adduser --system --no-create-home --uid 1001 --gid 1001 wav2vec2 && \
  chown -R wav2vec2:wav2vec2 /app

USER 1001:1001

COPY --chown=wav2vec2:wav2vec2 requirements.txt .

# remove torch as we want to install cpu-only linux wheel
RUN sed -i '/torch/d' requirements.txt

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"
RUN pip install --no-cache-dir -r requirements.txt && \
  pip install https://download.pytorch.org/whl/cpu/torch-1.10.1%2Bcpu-cp39-cp39-linux_x86_64.whl && \
  python -c "from os import getenv; from transformers import Wav2Vec2Processor; Wav2Vec2Processor.from_pretrained(getenv('model'));"

COPY --chown=wav2vec2:wav2vec2 onnx-inference.py onnx.py
COPY --chown=wav2vec2:wav2vec2 wav2vec2.onnx .

CMD ["uvicorn", "onnx:app", "--host", "0.0.0.0", "--port", "8000"]