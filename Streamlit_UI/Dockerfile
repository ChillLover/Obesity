FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY user_inter.py .

CMD ["streamlit", "run", "user_inter.py", "--server.port=8501", "--server.address=0.0.0.0"]