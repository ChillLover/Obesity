import streamlit as st
import pandas as pd
import requests


st.title("Предсказание ожирения")

input_data = st.file_uploader("Выбрать данные", type="csv")

if input_data:
    data = pd.read_csv(input_data)
    st.write("Содержимое файла: ", data)

    if st.button("Получить предсказания"):
        data = data.drop("Obesity", axis=1).to_dict(orient="records")

        response = requests.post("http://api:8000/predict_features", json={"data": data})

        if response.status_code == 200:
            preds = pd.read_json(response.json()["Answer"], orient="records")
            preds.rename(columns={0: "Предсказанные значения"}, inplace=True)
            st.write(preds)
        
        else:
            st.error(f"Ошибка API: {response.text}")
