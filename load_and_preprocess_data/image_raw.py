import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt


AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
img_height = 180
img_width = 180

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname='flower_photos', untar=True)

print(data_dir)


data_dir = pathlib.Path(data_dir)

print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

print(list_ds)

list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

print(list_ds)

for f in list_ds.take(5):
    print(f.numpy())

class_names = np.array(sorted(
    [item.name for item in data_dir.glob("*") if item.name != "LICENSE.txt"]))

print(class_names)

val_size = int(image_count * 0.2)

train_ds = list_ds.skip(val_size)

val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(train_ds))

# plt.figure(figsize=(10, 10))

# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(image_batch[i].numpy().astype('uint8'))
#     label = label_batch[i]
#     plt.title(class_names[label])
#     plt.axis("off")
# plt.show()


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
    validation_data=val_ds,
    epochs=5
)


ypreds = model.predict(val_ds)

res = list(map(lambda arr: np.argmax(arr), ypreds))

top_ten_val_res = np.array(res)
top_ten_val_act = np.concatenate([y for x, y in val_ds], axis=0)

result = np.logical_xor(top_ten_val_act, top_ten_val_res)

result_false = result == False
result_true = result == True

print("Correct Predictions: ", len(result[result_false]))
print("Incorrect Predictions: ", len(result[result_true]))