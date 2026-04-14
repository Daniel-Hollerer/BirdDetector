FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY model ./model

EXPOSE 5000

CMD ["python", "backend/main.py"]
