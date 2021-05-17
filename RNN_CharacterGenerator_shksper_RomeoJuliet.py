from tensorflow.keras.preprocessing import sequence
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

path_to_file=tf.keras.utils.get_file('Shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text=open(path_to_file,'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
print(text[:250])

vocab=sorted(set(text))

char2idx={u:i for i,u in enumerate(vocab)}
idx2char=np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int=text_to_int(text)
print("Text:",text[:13])
print("Encoded:",text_to_int(text[:13]))

def int_to_text(ints):
    try:
        ints=ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])
print(int_to_text(text_as_int[:13]))

seq_length=100
examples_per_epoch=len(text)//(seq_length+1)

char_dataset= tf.data.Dataset.from_tensor_slices(text_as_int)

sequences=char_dataset.batch(seq_length+1,drop_remainder=True)

def split_input_target(chunk):
    input_text=chunk[:-1]
    target_text=chunk[1:]
    return input_text,target_text
dataset=sequences.map(split_input_target)

for x,y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_text(x))
    print("\nOUTPUT")
    print(int_to_text(y))

    






