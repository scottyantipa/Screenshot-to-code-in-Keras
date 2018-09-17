import os
from flask import Flask, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from predict import predict

app = Flask(__name__)

@app.route('/', methods=["POST"])
def handle_image():
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not filename.find('.png'):
        return BadRequest("Invalid file type")

    # TODO DANGER parallel requests will conflict if they have the same file name
    # TODO Delete image after using it, or try not to store image at all
    input_dir = './images/'
    input_filepath = os.path.join(input_dir, filename)
    output_filepath = os.path.join('/output/', filename)
    input_file.save(input_filepath)

    prediction = predict(input_dir, filename)
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0')
