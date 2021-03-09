import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


imdb_dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz",
                                  imdb_dataset_url,
                                  untar=True,
                                  cache_dir=".",
                                  cache_subdir=""
                                  )

# print(os.path.dirname(dataset))

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# remove unsup directory
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


# Preprocessing

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

# print(type(raw_train_ds))

# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(3):
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])

# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])
