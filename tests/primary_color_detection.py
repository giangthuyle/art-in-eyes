import numpy as np
import webcolors
from PIL import Image


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def get_primary_color(image, n):
    m = np.sum(np.asarray(image), -1) < 255 * 3
    m = m / np.sum(np.sum(m))
    dx = np.sum(m, 0)
    dy = np.sum(m, 1)
    cx = np.sum(dx * np.arange(image.size[0]))
    cy = np.sum(dy * np.arange(image.size[1]))

    immat = image.load()
    return get_colour_name(immat[int(cx), int(cy)])


def crop_image(image, xmin, ymin, xmax, ymax):
    return Image.open(image).crop((xmin, ymin, xmax, ymax))
