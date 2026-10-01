FROM python:3.11-slim
WORKDIR /app
COPY . /app/
RUN pip install dash numpy==1.24.3 scikit-learn==1.3.0 cloudpickle matplotlib
EXPOSE 8050
CMD ["python", "app.py"]