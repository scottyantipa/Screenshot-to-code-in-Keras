#!/usr/bin/env python
# coding: utf-8

from os import listdir
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as KerasBackend
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm
import numpy as np
import h5py as h5py
from Bootstrap.compiler.classes.Compiler import *
import cv2


# https://github.com/tonybeltramelli/pix2code/blob/master/model/classes/Utils.py
class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")


# Read a file and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
max_length = 48 # No idea why they picked this
def generate_desc(model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        print(word + ' ', end='')
        if word == '<END>':
            break
    return in_text

def predict(input_path, file_name):
    # Initialize the function to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts([load_doc('./Bootstrap/bootstrap.vocab')])

    # From convert_imgs_to_arrays.py
    IMAGE_SIZE = 256
    output_path = './'

    # Im doing this weird dance of pushing into an array, and using np.array just because test_model_accuracy
    # does the same thing, and for some reason when I don't the features are slightly different
    images = []
    file_name_no_ex = file_name[:file_name.find(".png")]
    img = Utils.get_preprocessed_img("{}/{}".format(input_path, file_name), IMAGE_SIZE)
    np.savez_compressed("{}/{}".format(output_path, file_name_no_ex), features=img)
    image = np.load("{}/{}.npz".format(output_path, file_name_no_ex))
    images.append(image['features'])
    images = np.array(images, dtype=float)

    #load model and weights
    json_file = open('/data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/data/weights.h5")

    desc = generate_desc(loaded_model, tokenizer, images[0], max_length)
    predicted = desc.split() # is this correct? Taken from line 11 evaluate_model

    # Compile the tokens into HTML and css
    dsl_path = "./Bootstrap/compiler/assets/web-dsl-mapping.json"
    compiler = Compiler(dsl_path)
    compiled_website = compiler.compile(predicted, 'output.html')

    # Cleanup keras betweeen requests https://github.com/RasaHQ/rasa_core/issues/80
    KerasBackend.clear_session()

    return compiled_website
