import torch
from PIL import Image

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    for f in ['zidane.jpg', 'bus.jpg']:
        torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)

    img1 = Image.open('zidane.jpg')
    img2 = Image.open('bus.jpg')
    imgs = [img1, img2]

    results = model(imgs, size=640)
    results.print()
