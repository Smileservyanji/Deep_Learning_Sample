import tensorflow as tf
from tensorflow import keras

# Check the available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define your model inside the strategy scope
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Load your data and create datasets (train and validation)
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0  # Normalize pixel values to [0, 1]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)

# Train your model using fit
model.fit(train_dataset, epochs=5, validation_data=val_dataset)
