from os import listdir
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Read a document and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def make_tokenizer():
    # Initialize the function that will create our vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)

    # Load all the HTML files
    X = []
    all_filenames = listdir('HTML/resources/html/')
    all_filenames.sort()
    for filename in all_filenames:
        X.append(load_doc('HTML/resources/html/'+filename))

    # Create the vocabulary from the html files
    tokenizer.fit_on_texts(X)
    return tokenizer, X

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(900):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-100:]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # Print the prediction
        print(' ' + word, end='')
        # stop if we predict the end of the sequence
        if word == 'END':
            break
    return

# - Load the model
# - get the image to run through the network
# - run the network
# - tokenize to html
def evaluate():
    model = load_model("/data/model.h5")

    tokenizer, X = make_tokenizer()

    # Add +1 to leave space for empty words
    vocab_size = len(tokenizer.word_index) + 1
    # Translate each word in text file to the matching vocabulary index
    sequences = tokenizer.texts_to_sequences(X)
    # The longest HTML file
    max_length = max(len(s) for s in sequences)

    # load image and create html
    print("[] Creating InceptionResNetV2 ...")
    weights_dir = '/data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    IR2 = InceptionResNetV2(weights=weights_dir, include_top=False)
    print("[x] Created InceptionResNetV2")

    test_image = img_to_array(load_img('HTML/resources/images/sticky.jpg', target_size=(299, 299)))
    test_image = np.array(test_image, dtype=float)
    test_image = preprocess_input(test_image)
    test_features = IR2.predict(np.array([test_image]))

    print("[] Generating desc...")
    desc = generate_desc(model, tokenizer, np.array(test_features), 100)
    print("[x] Generated desc")
    print(desc)

if __name__ == '__main__':
    evaluate()
