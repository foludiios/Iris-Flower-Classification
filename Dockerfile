FROM python:3.11

WORKDIR /irispred

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.Api:app", "--host", "0.0.0.0", "--port", "8000"]