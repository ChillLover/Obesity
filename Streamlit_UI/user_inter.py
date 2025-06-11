import streamlit as st
import pandas as pd
import requests


st.title("Предсказание ожирения")

input_data = st.file_uploader("Выбрать данные", type="csv")

if input_data:
    data = pd.read_csv(input_data)
    st.write("Содержимое файла: ", data)

    if st.button("Получить предсказания"):
        data = data.to_dict(orient="records")

        response = requests.post("http://api-obesity/predict_features", json={"data": data})

        if response.status_code == 200:
            preds = pd.read_json(response.json()["Answer"], orient="records")
            preds.rename(columns={0: "Predictions"}, inplace=True)
            result = pd.concat([preds, pd.DataFrame.from_dict(data, orient="columns")], axis=1)

            result["Predictions"] = result["Predictions"].map({
                0: "Insufficient_Weight",
                1: "Normal_Weight",
                2: "Overweight_Level_I",
                3: "Overweight_Level_II",
                4: "Obesity_Type_I",
                5: "Obesity_Type_II",
                6: "Obesity_Type_III"
            })

            st.write(result)

            st.download_button("Скачать предсказания", data=result.to_csv(index=False), file_name="preds.csv")
        
        else:
            st.error(f"Ошибка API: {response.text}")
