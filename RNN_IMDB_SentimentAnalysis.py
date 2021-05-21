import keras.preprocessing.text
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os
import numpy as np

VOCAB_SIZE=88000
MAXLEN=250
BATCH_SIZE=64

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=VOCAB_SIZE)

# train_data[0]
# len(train_data[0])

train_data=sequence.pad_sequences(train_data,MAXLEN)
test_data=sequence.pad_sequences(test_data,MAXLEN)
# train_data[1]

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

# model.summary()

model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['acc'])
history=model.fit(train_data,train_labels,epochs=10,validation_split=0.2)

results=model.evaluate(test_data,test_labels)
# print(results)

word_index=imdb.get_word_index()

def encode_text(text):
    tokens=keras.preprocessing.text.text_to_word_sequence(text)
    tokens=[word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens],MAXLEN)[0]

text="that movie was so amazing, so amazing"
encoded=encode_text(text)
print(encoded)

reverse_word_index={value:key for (key,value) in word_index.items()}

def decode_integers(integers):
    PAD=0
    text=""
    for num in integers:
        if num!=PAD:
            text+=reverse_word_index[num]+""

    return text[:-1]
print(decode_integers(encoded))

def predict(text):
    encoded_text=encode_text(text)
    pred=np.zeros((1,250))
    pred[0]=encoded_text
    result=model.predict(pred)
    print(result[0])

positive_review="That movie was so awesome!, I really love it and would watch it again because it was great"
predict(positive_review)

negative_review="that movie shocked. I hated it and wouldn't watch it again. was one of the worst movie watched"
predict(negative_review)