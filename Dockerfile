# Python and package versions have to match the ones used to pickle models/model.joblib
FROM python:3.13-slim

WORKDIR /submission

COPY requirements.txt /submission/requirements.txt
RUN pip install --no-cache-dir -r /submission/requirements.txt

COPY features.py evaluate.py submission.py /submission/
COPY models/model.joblib /submission/models/model.joblib

ENV PYTHONPATH=/submission
