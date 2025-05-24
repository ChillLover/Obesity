FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY user_inter.py .

CMD ["streamlit", "run", "user_inter.py", "--server.port=8501", "--server.address=0.0.0.0"]