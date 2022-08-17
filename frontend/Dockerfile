FROM python:3.8

EXPOSE 8501
EXPOSE 8000

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . .
WORKDIR /app

# ENTRYPOINT ["streamlit", "run", "main.py"]
ENTRYPOINT ["streamlit", "run", "main.py", "–server.port=8501", "–server.address=0.0.0.0"]