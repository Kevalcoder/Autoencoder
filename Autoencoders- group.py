# Keval and Smit group assignment
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from tensorflow.python.ops.gen_array_ops import shape
from sklearn.model_selection import KFold
import numpy as np


dataset = fetch_olivetti_faces(random_state=42)

dataset.keys()

X = dataset.images

X.shape
y = dataset.target
X = X.reshape(400, 4096)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_copy = X.copy()

pca = PCA(n_components=256)
X_pca = pca.fit_transform(X)

X_pca = X_pca.reshape(400, 16, 16, 1)

results = []
kfold = KFold(n_splits=3, shuffle=True)

learning_rate = [0.1, 0.5, 1]
encoding_layes = [64, 128, 164]

count = 0
for train, test in kfold.split(X_pca, y):

    temp = {}
    encoder_input = keras.Input(shape=(16, 16, 1), name="Input")
    x = keras.layers.Flatten()(encoder_input)

    for j in range(len(encoding_layes)):
        encoder_output = keras.layers.Dense(
            encoding_layes[j], activation="relu")(x)

        encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        # decoder
        decoder_input = keras.layers.Dense(
            256, activation="relu")(encoder_output)

        decoder_output = keras.layers.Reshape((16, 16, 1))(decoder_input)

        autoencoder = keras.Model(
            encoder_input, decoder_output, name="autoencoder")

        for l in range(len(learning_rate)):
            optimizer = keras.optimizers.Adam(lr=learning_rate[l])

            autoencoder.compile(optimizer, loss="mse")

            history = autoencoder.fit(
                X_pca[train], X_pca[train], epochs=20, batch_size=50)

            temp["history"] = history.history

            count += 1

            temp["learning_rate"] = l

            temp["encoding_layers"] = j

            results.append(temp)


len(results)

best_model = []
best_params_index = []
minimum = []
for i in range(len(results)):

    best_params_index.append(np.argmin(results[i]["history"]["loss"]))

for j in range(len(best_params_index)):
    minimum.append(np.min(results[j]["history"]["loss"]))

np.min(minimum)
best_index = np.argmin(minimum)

learning_rate_index = results[best_index]["learning_rate"]
learning_rate_index

learning_rate = learning_rate[learning_rate_index]
learning_rate

encoding_layers_index = results[best_index]["encoding_layers"]
encoding_layers = encoding_layes[encoding_layers_index]
encoding_layers

# loss at this configuration is
model_loss = np.min(results[best_index]["history"]["loss"])
model_loss
