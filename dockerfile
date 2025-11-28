FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git libgl1 build-essential && apt-get clean

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "smartta_unified/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
