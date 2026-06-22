FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install streamlit requests dotenv openai numpy pydub modal
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
