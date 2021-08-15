import torch
from flask import Flask

APP_NAME = 'The Eye'
model = None


def init_foundation():
    # global model
    # model = torch.hub.load(
    #     "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    # ).autoshape()  # force_reload = recache latest code
    # model.eval()
    pass


def create_app(config_file: str):
    the_app = Flask(APP_NAME)
    the_app.config.from_pyfile(config_file)

    init_foundation()
    return the_app
