import operator

import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageOps

imported = tf.saved_model.load('../converted_savedmodel/model.savedmodel')
model = imported.signatures['serving_default']

labels = ['41541', '54563', '75795', '51578', '101788']


def recognize(image):
    image_mat = np.asarray(image)
    normalized_image_arr = (image_mat.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_arr

    predictions = model(sequential_9_input=data)['sequential_11'][0].numpy()
    result = dict(zip(labels, predictions))
    best_match = max(result.items(), key=operator.itemgetter(1))[0]
    df = pd.read_csv('paintings.csv')
    match = df[df['new_filename'] == f'{best_match}.jpg'].iloc[0]
    return {
        'artist': match.artist,
        'genre': match.genre,
        'style': match.style,
        'date': match.date
    }


def change_type_info():
    df = pd.read_excel('../infor-paintings.xlsx')
    df.to_csv('paintings.csv')


if __name__ == '__main__':
    image = Image.open('../1374px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    # change_type_info()
    print(recognize(image))
