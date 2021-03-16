import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

print(tf.__version__)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname='flower_photos', untar=True)

print(data_dir)


data_dir = pathlib.Path(data_dir)

print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

print(image_count)


roses = list(data_dir.glob('roses/*'))

print(len(roses))

image = PIL.Image.open(str(roses[0]))

# np_image = np.asarray(image)

# print(np_image.shape)


batch_size = 32
img_height = 180
img_width = 180


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


print(train_ds.class_names)

class_names = train_ds.class_names


# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(class_names[labels[i]])
#         plt.axis('off')
# plt.show()

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 5

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

h = model.fit(train_ds,
    validation_data=validation_ds,
    epochs=5
)

ypreds = model.predict(validation_ds)

res = list(map(lambda arr: np.argmax(arr), ypreds))

top_ten_val_res = np.array(res)
top_ten_val_act = np.concatenate([y for x, y in validation_ds], axis=0)

result = np.logical_xor(top_ten_val_act, top_ten_val_res)

result_false = result == False
result_true = result == True

print("Correct Predictions: ", len(result[result_false]))
print("Incorrect Predictions: ", len(result[result_true]))

# print(result)