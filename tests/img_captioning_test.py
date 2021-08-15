import pickle

import numpy as np
import tensorflow as tf

# Config
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = 5001
attention_features_shape=64

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layers = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layers)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.weight_1 = tf.keras.layers.Dense(units)
        self.weight_2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, state):
        state_with_time = tf.expand_dims(state, 1)  # (batch_size x 1 x hidden_size)

        attention_hidden_layer = (
            tf.nn.tanh(self.weight_1(features) + self.weight_2(state_with_time)))  # (batch_size x 64 x 256)

        score = self.V(attention_hidden_layer)  # (batch_size x 64 x 1)

        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size x 64 x 1)

        context_vector = attention_weights * features  # (batch_size x hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)  # fully connected layers

    def call(self, image_features):
        x = self.fc(image_features)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, return_value, features, current_state):
        context_vector, attention_weights = self.attention(features, current_state)

        return_value = self.embedding(return_value)

        return_value = tf.concat([tf.expand_dims(context_vector, 1), return_value], axis=-1)

        output, state = self.gru(return_value)

        return_value = self.fc1(output)

        return_value = tf.reshape(return_value, (-1, return_value.shape[2]))

        return_value = self.fc2(return_value)

        return return_value, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()
with open('/home/danielng/Projects/the-eye/tokenizer.pickle', 'rb') as saved_model:
    tokenizer = pickle.load(saved_model)

checkpoint_path = '../checkpoints/train'
checkpoint = tf.train.Checkpoint(
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer
)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
checkpoint.restore(manager.latest_checkpoint)


def calc_max_length(captions):
    return max(len(c) for c in captions)


max_length = 20


def load_image(image_path: str):
    img = tf.io.read_file(image_path)  # binary
    img = tf.image.decode_jpeg(img, channels=3)  # matrix RBG
    img = tf.image.resize(img, (299, 299))  # ( 299 x 299 x 3)
    # Xu li anh de phu hop voi model Inception v3
    img = tf.keras.applications.inception_v3.preprocess_input(img)  # value (-1, 1)
    return img, image_path


def evaluate(image_path: str):
    img = load_image(image_path)[0]

    # mo them 1 chieu khong gian de ki hieu so luong anh se chay qua inception ( 299 x 299 x 3) => (n x 299 x 299 x 3)
    temp_input = tf.expand_dims(img, 0)
    img_tensor_val = image_features_extract_model(temp_input)  # (n x 8 x 8 x 2048)
    img_tensor_val = tf.reshape(img_tensor_val,
                                (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  # ( n x 64 x 2048)

    features = encoder(img_tensor_val)
    description_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)  # 1 x 1
    result = []

    # reset decoder state
    new_decoder_state = decoder.reset_state(batch_size=1)
    attention_plot = np.zeros((max_length, attention_features_shape))

    for i in range(max_length):
        predictions, state, weights = decoder(description_input, features, new_decoder_state)

        attention_plot[i] = tf.reshape(weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        description_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def describe_image(img_url: str, unique_name: str, extension='.jpg'):
    image_path = tf.keras.utils.get_file(unique_name + extension, origin=img_url)

    # evaluate/ predict caption for image
    result, plot = evaluate(image_path)
    print('Prediction: ', ' '.join(result))


if __name__ == '__main__':
    url = 'https://scontent-hkt1-1.xx.fbcdn.net/v/t1.18169-9/22789015_10211284501246581_8116701730954350879_n.jpg?_nc_cat=111&ccb=1-4&_nc_sid=09cbfe&_nc_ohc=XQkxgNVAkTUAX-MDIw6&_nc_ht=scontent-hkt1-1.xx&oh=3ebf74d9a605b5377062f0c7a989eec8&oe=613DF5BA'
    image_path = tf.keras.utils.get_file('avatar.jpg.jpg', origin=url)
    describe_image(url, 'avatar.jpg', '.jpg')
