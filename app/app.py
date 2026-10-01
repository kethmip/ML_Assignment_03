from dash import Dash, html, dcc, Input, Output, State
import cloudpickle
import numpy as np

app = Dash(__name__)

# Load A3 model and preprocessing files
with open("best_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

with open("scaler_a3.pkl", "rb") as f:
    scaler = cloudpickle.load(f)

with open("label_encoder_a3.pkl", "rb") as f:
    le = cloudpickle.load(f)

brands = list(le.classes_)

app.layout = html.Div([

    html.H1("Car Price Class Prediction"),

    html.P("Enter the car information below to predict its selling price class (0, 1, 2, or 3)."),

    html.Label("Max Power"),
    dcc.Input(id="max_power", type="number"),

    html.Br(),
    html.Br(),

    html.Label("Transmission"),
    dcc.Dropdown(
        id="transmission",
        options=[
            {"label": "Manual", "value": 0},
            {"label": "Automatic", "value": 1}
        ]
    ),

    html.Br(),

    html.Label("Brand"),
    dcc.Dropdown(
        id="brand",
        options=[{"label": brand, "value": brand} for brand in brands]
    ),

    html.Br(),

    html.Button("Predict Class", id="predict_button", n_clicks=0),

    html.Br(),
    html.Br(),

    html.Div(id="prediction_output")
])


@app.callback(
    Output("prediction_output", "children"),
    Input("predict_button", "n_clicks"),
    State("max_power", "value"),
    State("transmission", "value"),
    State("brand", "value")
)

def predict_class(n_clicks, max_power, transmission, brand):

    if n_clicks > 0:

        if max_power is None or transmission is None or brand is None:
            return "Please enter Max Power, Transmission, and Brand."

        # Scale max_power using the same scaler used during training
        max_power_scaled = scaler.transform([[max_power]])[0][0]

        # Encode brand using the same LabelEncoder used during training
        brand_encoded = le.transform([brand])[0]

        # Model expects: intercept, max_power, transmission, brand
        sample = np.array([[1, max_power_scaled, transmission, brand_encoded]])

        prediction = model.predict(sample)[0]

        return f"Predicted Selling Price Class: {prediction}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)