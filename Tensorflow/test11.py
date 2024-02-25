import tensorflow as tf
import tensorflow_datasets as tfds

# Define the GPU device to use (e.g., GPU:0)
gpu_device = "/gpu:0"

# Wrap the training code in a tf.device context to specify the GPU
with tf.device(gpu_device):
    # Load the CIFAR-10 dataset
    dataset_name = "cifar10"
    (ds_train, ds_test), ds_info = tfds.load(
        name=dataset_name,
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,  # Load data in (image, label) format
        with_info=True,
    )

    # Print dataset info
    print("Dataset info:")
    print(ds_info)

    # Define preprocessing functions (e.g., resize, normalize)
    def preprocess_image(image, label):
        image = tf.image.resize(image, [32, 32])  # Resize to a consistent size
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
        return image, label

    # Apply preprocessing to the dataset
    ds_train = ds_train.map(preprocess_image)
    ds_test = ds_test.map(preprocess_image)

    # Batch and shuffle the training dataset
    batch_size = 64
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.shuffle(buffer_size=10000)

    # Batch the testing dataset
    ds_test = ds_test.batch(batch_size)

    # Define a model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10),
    ])

    # Compile the model
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # Train the model
    epochs = 10
    model.fit(ds_train, epochs=epochs)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

