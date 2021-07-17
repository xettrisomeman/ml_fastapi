FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . /app/

EXPOSE 8000


