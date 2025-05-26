FROM tiangolo/uwsgi-nginx-flask:python3.9

COPY ./requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY ./app /app/app
COPY ./static /app/static
COPY ./templates /app/templates

# The tiangolo image expects the Flask app in /app/main.py or /app/app/main.py

