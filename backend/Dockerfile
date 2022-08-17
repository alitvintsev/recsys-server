# 
FROM python:3.9

EXPOSE 8000

# 
WORKDIR /server

# 
COPY requirements.txt ./server/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r ./server/requirements.txt
# RUN pip install -U pip
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY . .

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

