import gradio as gr
import joblib
import numpy as np

print("Starting Gradio App...")

model = joblib.load("model.pkl")

def predict(stock1, stock2, stock3, stock4, day, month):

    data = np.array([[stock1, stock2, stock3, stock4, day, month]])

    prediction = model.predict(data)

    return float(prediction[0])


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Stock 1"),
        gr.Number(label="Stock 2"),
        gr.Number(label="Stock 3"),
        gr.Number(label="Stock 4"),
        gr.Number(label="Day"),
        gr.Number(label="Month")
    ],
    outputs="number",
    title="Stock Price Prediction"
)

interface.launch()