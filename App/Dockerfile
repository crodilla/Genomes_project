FROM python:3.11


WORKDIR /app

COPY . /app
# RUN apt-get update && apt-get install -y build-essential

RUN python3.11 -m pip install -r requirements.txt
EXPOSE 80


CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]