# ── Base image ──
FROM python:3.11-slim

# tesseract-ocr     : OCR engine used by pytesseract
# tesseract-ocr-urd : Urdu script recognition (GCC/expat-labor documents)
# tesseract-ocr-ara : Arabic script recognition (GCC official documents)
# poppler-utils     : used by pdf2image to convert PDF pages to images
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-urd \
    tesseract-ocr-ara \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional AI/NER model (see requirements.txt) — comment out to keep the
# image smaller if you don't need free-text name/org/location detection.
RUN python -m spacy download en_core_web_sm || true

COPY . .

# Render (and most PaaS platforms) inject $PORT at runtime.
ENV PORT=8080
EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 180
