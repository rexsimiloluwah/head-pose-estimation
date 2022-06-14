FROM python:3.7.3-stretch

WORKDIR /app

COPY . . /app/ 

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt 

EXPOSE 8501 

RUN ["streamlit","run","app.py"]