import opendatasets as od
import pandas

od.download(
    "https://www.kaggle.com/competitions/arabic-manuscripts-digitization/data")

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define paths
data_dir = '/content/arabic-manuscripts-digitization/'
train_image_dir = os.path.join(data_dir, 'train', 'train')
test_image_dir = os.path.join(data_dir, 'test', 'test')
train_csv = os.path.join(data_dir, 'train_df.csv')
test_csv = os.path.join(data_dir, 'test_df.csv')

# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Step 1: Remove missing images
def check_images(df, img_dir):
    return [img for img in df['image'] if not os.path.exists(os.path.join(img_dir, img))]

missing_train = check_images(train_df, train_image_dir)
train_df = train_df[~train_df['image'].isin(missing_train)]
print(f"Train images after removal: {len(train_df)}")

# Step 2: Constants
IMG_HEIGHT = 32
IMG_WIDTH = 256
MAX_LABEL_LEN = 64

# Step 3: Preprocess Images
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    new_w = int(w * IMG_HEIGHT / h)
    img = cv2.resize(img, (new_w, IMG_HEIGHT))
    if new_w < IMG_WIDTH:
        pad = np.zeros((IMG_HEIGHT, IMG_WIDTH - new_w), dtype=np.uint8)
        img = np.hstack((img, pad))
    else:
        img = img[:, :IMG_WIDTH]
    img = img / 255.0
    return img

# Step 4: Encode Text
all_chars = sorted(set(''.join(train_df['text'].astype(str))))
char_to_index = {c: i for i, c in enumerate(all_chars)}
index_to_char = {i: c for c, i in char_to_index.items()}
num_classes = len(char_to_index) + 1  # +1 for CTC blank

def encode_label(text):
    encoded = [char_to_index[c] for c in text if c in char_to_index]
    length = len(encoded)
    if length < MAX_LABEL_LEN:
        encoded += [0] * (MAX_LABEL_LEN - length)
    else:
        encoded = encoded[:MAX_LABEL_LEN]
        length = MAX_LABEL_LEN
    return encoded, length

# Step 5: Load Training Data
train_images = []
train_labels = []
train_label_lengths = []

for _, row in train_df.iterrows():
    img = preprocess_image(os.path.join(train_image_dir, row['image']))
    if img is not None:
        encoded, length = encode_label(row['text'])
        train_images.append(img)
        train_labels.append(encoded)
        train_label_lengths.append(length)

train_images = np.array(train_images)[..., np.newaxis]
train_labels = np.array(train_labels, dtype=np.int32)
train_label_lengths = np.array(train_label_lengths, dtype=np.int32)
input_lengths = np.ones((len(train_images), 1)) * (IMG_WIDTH // 4)

# Step 6: Split Train/Val
split = int(0.8 * len(train_images))
train_data = {
    "input_img": train_images[:split],
    "labels": train_labels[:split],
    "input_length": input_lengths[:split],
    "label_length": train_label_lengths[:split][:, np.newaxis],
}
val_data = {
    "input_img": train_images[split:],
    "labels": train_labels[split:],
    "input_length": input_lengths[split:],
    "label_length": train_label_lengths[split:][:, np.newaxis],
}
train_ds = tf.data.Dataset.from_tensor_slices((train_data, np.zeros((split,)))).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_data, np.zeros((len(train_images)-split,)))).batch(32).prefetch(tf.data.AUTOTUNE)

# Step 7: Build CTC Model
def build_ctc_model(input_shape, num_classes, max_label_len):
    input_img = layers.Input(shape=input_shape, name='input_img')
    labels = layers.Input(shape=(max_label_len,), dtype='int32', name='labels')
    input_length = layers.Input(shape=(1,), dtype='int32', name='input_length')
    label_length = layers.Input(shape=(1,), dtype='int32', name='label_length')

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 1))(x)

    b, h, w, c = x.shape
    x = layers.Reshape((w, h * c))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    y_pred = layers.Dense(num_classes, activation='softmax', name='y_pred')(x)

    def ctc_lambda(args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name='ctc_loss')(
        [y_pred, labels, input_length, label_length])

    model = models.Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    pred_model = models.Model(inputs=input_img, outputs=y_pred)

    return model, pred_model

ctc_model, prediction_model = build_ctc_model((IMG_HEIGHT, IMG_WIDTH, 1), num_classes, MAX_LABEL_LEN)
ctc_model.compile(optimizer='adam', loss={'ctc_loss': lambda y_true, y_pred: y_pred})


# Step 8: Train
history = ctc_model.fit(train_ds, validation_data=val_ds, epochs=50)


# Step 9: Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CTC Loss Over Epochs")
plt.show()

# Step 10: Prediction on Test Data
test_df = pd.read_csv(test_csv)
test_imgs = []
img_names = []

for img_name in test_df['image']:
    path = os.path.join(test_image_dir, img_name)
    img = preprocess_image(path)
    if img is not None:
        test_imgs.append(img)
        img_names.append(img_name)

test_imgs = np.array(test_imgs)[..., np.newaxis]
preds = prediction_model.predict(test_imgs)
decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1])

# Step 11: Convert predictions to text
results = []
for seq in decoded[0].numpy():
    text = ''.join([index_to_char.get(i, '') for i in seq if i != -1])
    results.append(text)

# Step 12: Create submission
submission_df = pd.DataFrame({
    'image': img_names,
    'text': results
})
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission saved to 'submission.csv'")
