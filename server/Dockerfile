FROM python:3.10-slim-buster

ENV APP_DIR "./"
RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

# Install dependencies
RUN apt-get update
RUN apt-get install -y gcc g++ libpng-dev libjpeg-dev
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONPATH $APP_DIR

CMD [ "flask", "--app", "./app.py", "run", "-h", "0.0.0.0" ]