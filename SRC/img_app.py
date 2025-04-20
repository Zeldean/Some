import dash
from dash import dcc, html, Input, Output, State
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image

# === Model + Settings ===
model = load_model("../Artifacts/imageprediction.h5")
image_size = (80, 60)  # (width, height)

# === Dash App Setup ===
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Image Prediction App"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and drop or click to select an image']),
        style={
            'width': '60%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '20px auto'
        },
        accept='image/*'
    ),
    html.Div(id='output-image'),
    html.Div(id='prediction-text')
])

def preprocess_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert('RGB')

    # Convert to array, resize, normalize
    img = np.array(img)
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.callback(
    Output('output-image', 'children'),
    Output('prediction-text', 'children'),
    Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is None:
        return "", ""

    img_array = preprocess_image(contents)
    prediction = model.predict(img_array)[0][0]
    label = "👛 Woman's Accessory" if prediction > 0.5 else "👞 Non-Woman Product"

    return html.Img(src=contents, style={'width': '200px'}), f"Prediction: {label} (Confidence: {prediction:.2f})"

if __name__ == '__main__':
    app.run(debug=True)
