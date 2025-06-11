import pandas as pd
import requests
import tempfile
import gradio as gr


def get_preds(data):
    data = data.name
    data = pd.read_csv(data)
    data = data.to_dict(orient="records")

    response = requests.post("http://api-obesity/predict_features", json={"data": data}) #api:8000

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
    
        temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        result.to_csv(temp_file.name, index=False)

        return result, temp_file.name                                                     


with gr.Blocks() as demo:
    input_file = gr.File(label="Загрузить файл", file_types=[".csv"])
    process_button = gr.Button("Обработать файл")
    output = gr.DataFrame(label="Полученные предсказания")
    download_button = gr.DownloadButton(label="Скачать предсказания")

    process_button.click(
        fn=get_preds,
        inputs=input_file,
        outputs=[output, download_button],
    )

demo.launch(server_name="0.0.0.0")
