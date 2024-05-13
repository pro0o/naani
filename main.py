from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

model_path = r'C:\Users\probi\Downloads\Naani\model.h5'
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_uri = request.json['image_uri']

        response = requests.get(image_uri)
        response.raise_for_status() 
        image_data = response.content

        test_image = image.load_img(BytesIO(image_data), target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.

        prediction = model.predict(test_image)

        if prediction[0][0] > 0.5:
            result = 'normal'
        else:
            result = 'cataract'

        return jsonify({'prediction': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
