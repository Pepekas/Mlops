FROM python:3.11-slim

# Установка системных зависимостей, необходимых для сборки pandas и других пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

COPY app.py cars.csv /

EXPOSE 8081

ENTRYPOINT ["python"]
CMD ["/app.py"]