from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import textwrap
from tensorflow.keras.models import load_model
from io import BytesIO
import base64

app = Flask(__name__)

# Load the pre-trained model
model = load_model('handwriting_model.h5')

def generate_handwriting_from_text(input_text, base_width=1000, base_line_height=60):
    """
    Generate a handwritten-style image from text input.

    Parameters:
        input_text (str): The input text to convert to handwriting.
        base_width (int): The width of the image canvas in pixels.
        base_line_height (int): The height of each line in pixels.

    Returns:
        np.ndarray: Handwritten-style image as a NumPy array.
    """
    max_chars_per_line = base_width // 12  # Calculate max characters per line
    wrapped_text = textwrap.wrap(input_text, width=max_chars_per_line)  # Wrap the text
    canvas_height = (base_line_height + 10) * len(wrapped_text) + 20  # Calculate canvas height

    # Create a blank white canvas
    blank_image = np.ones((canvas_height, base_width, 1), dtype=np.uint8) * 255

    # Add each line of text to the image
    for i, line in enumerate(wrapped_text):
        y_pos = (i + 1) * base_line_height - 10  # Calculate vertical position for each line
        cv2.putText(blank_image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    return blank_image

@app.route('/')
def index():
    """
    Render the HTML template for the frontend.
    """
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate a handwritten image from text.
    """
    input_text = request.form['text']  # Get the input text from the form
    if not input_text:
        return jsonify({'error': 'Text input is required'}), 400

    # Generate the handwritten image
    handwritten_image = generate_handwriting_from_text(input_text, base_width=1000, base_line_height=60)

    # Convert the image to PNG format and encode it as base64
    is_success, buffer = cv2.imencode('.png', handwritten_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the encoded image to the frontend
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
