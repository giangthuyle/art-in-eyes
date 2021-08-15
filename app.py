import requests
from PIL import Image, ImageOps
from flask import request, Response

from foundation import create_app
from tests.artwork_classification import recognize
from tests.primary_color_detection import crop_image, get_primary_color

app = create_app('/home/danielng/Projects/the-eye/config.py')


@app.route('/detect', methods=['POST'])
def detect_color():
    if request.files.get('image'):
        url = 'http://localhost:5000/v1/object-detection/yolov5s'
        resp = requests.post(url, stream=True, allow_redirects=False,
                             files=request.files)
        image = request.files.get('image')

        def generate():
            for chunk in resp.raw.stream(decode_content=False):
                yield chunk

        out = Response(generate(), headers={
            'Content-Type': 'application/json'
        })
        out.status_code = resp.status_code

        i = 0
        for img in out.json:
            xmin, ymin, xmax, ymax = img['xmin'], img['ymin'], img['xmax'], img['ymax']
            cropped_img = crop_image(image, xmin, ymin, xmax, ymax)
            # cropped_img.save(f'test{i}.jpg')
            # i += 1
            print(get_primary_color(cropped_img, 3)[1], img['name'])

        return out


@app.route('/info', methods=['POST'])
def info():
    if request.files.get('image'):
        image = Image.open(request.files['image'])
        image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
        return recognize(image)


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.files.get('image'):
        return app.config['UPLOAD_FOLDER']


if __name__ == '__main__':
    app.run(port=8000, debug=False, load_dotenv=True)
