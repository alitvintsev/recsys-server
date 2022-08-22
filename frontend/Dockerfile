FROM python:3.8

# ARG STREAMLIT_PATH="/root/.streamlit/"

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# RUN mkdir -p $STREAMLIT_PATH
# COPY config.toml $STREAMLIT_PATH

COPY . .

WORKDIR /app

EXPOSE 8501
EXPOSE 8000

ENTRYPOINT ["streamlit", "run", "main.py", "–server.port=8501", "–server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]