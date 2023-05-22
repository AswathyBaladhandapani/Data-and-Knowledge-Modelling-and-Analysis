# import required packages
from tensorflow.keras.utils import get_file
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tarfile
from glob import glob
import os,re,string

# loading data from the data folder
def loading_texts_labels(path, folders):
    texts,labels = [],[]
    for index,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(index)
    
    return texts, np.array(labels).astype(np.int64)

# Preprocessing the reviews
def rev_preprocess(reviews):
    tokens = re.compile("[.;:!#\'?,\"()\[\]]|(<br\s*/><br\s*/>)|(\-)|(\/)")
    
    return [tokens.sub("", line.lower()) for line in reviews]

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model
    my_model = keras.models.load_model('./models/Group26_NLP_model.h5')

	# 2. Load your testing data
    path_loc ='./data/aclImdb/'
    review = ['neg','pos']
    X_test,Y_test = loading_texts_labels(f'{path_loc}test',review)
    X_train,Y_train = loading_texts_labels(f'{path_loc}train',review)
    
    cleaned_X_train = rev_preprocess(X_train)
    cleaned_X_test = rev_preprocess(X_test)
    
    # tokenizer
    # converting tokens into sequence
    token = keras.preprocessing.text.Tokenizer()
    token.fit_on_texts(cleaned_X_train)
    X_test_tok = token.texts_to_sequences(cleaned_X_test)
    
    X_test_tok = keras.preprocessing.sequence.pad_sequences(X_test_tok,padding='post',maxlen=1000)

	# 3. Run prediction on the test data and print the test accuracy
    test_values = my_model.evaluate(X_test_tok,Y_test)
    test_acc = test_values[1]
    print('test accuracy:',test_acc*100)