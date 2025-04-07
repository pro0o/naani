FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN ollama run llama3

EXPOSE 5000

ENV FLASK_APP main.py

CMD ["flask", "run", "--host=0.0.0.0"]

#python main.py