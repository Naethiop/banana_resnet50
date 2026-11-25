FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BANANA_ENABLE_HISTORY=false

WORKDIR /app

COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY src ./src
COPY wsgi.py ./wsgi.py
COPY models/resnet50_banana.keras models/class_names.json models/model_meta.json models/input_check_bank.npz ./models/

EXPOSE 8080
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "300", "--bind", "0.0.0.0:8080", "wsgi:app"]
