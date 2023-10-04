FROM python:3.11.5-slim-bookworm
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT uvicorn main:app --host 0.0.0.0
