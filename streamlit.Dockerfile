FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit requests dotenv openai numpy
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
