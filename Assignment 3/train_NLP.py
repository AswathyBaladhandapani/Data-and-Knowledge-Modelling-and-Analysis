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
	# 1. load your training data
    path_loc ='./data/aclImdb/'
    review = ['neg','pos']
    X_train,Y_train = loading_texts_labels(f'{path_loc}train',review)
    cleaned_X_train = rev_preprocess(X_train)
    
    # tokenizer
    # converting tokens into sequence
    token = keras.preprocessing.text.Tokenizer()
    token.fit_on_texts(cleaned_X_train) 
    X_train_tok = token.texts_to_sequences(cleaned_X_train)
    
    # splitting the training data into train and validation set
    X_train_tok = keras.preprocessing.sequence.pad_sequences(X_train_tok,padding='post',maxlen=1000)
    X_train, X_val, y_train, y_val = train_test_split(X_train_tok, Y_train, test_size=0.3, random_state=25)
    
	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
    # Building the Model for training the data
    # input shape - word count in reviews
    word_length = len(token.word_index)+1

    my_model = keras.Sequential()
    my_model.add(keras.layers.Embedding(word_length, 16))
    my_model.add(keras.layers.Dropout(0.2))
    my_model.add(keras.layers.Conv1D(filters=16,kernel_size=2,padding='valid',activation='relu'))
    my_model.add(keras.layers.GlobalAveragePooling1D())
    my_model.add(keras.layers.Dropout(0.1))
    my_model.add(keras.layers.Dense(32, activation='relu'))
    my_model.add(keras.layers.Dense(1, activation='sigmoid'))
    my_model.summary()
    
    my_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']
    model_history = my_model.fit(X_train,y_train,epochs=25,validation_data=(X_val, y_val),verbose=1,batch_size=512)
    
    print('training accuracy:',model_history.history['acc'][-1]*100)
                     
	# 3. Save your model
    my_model.save('./models/Group26_NLP_model.h5')