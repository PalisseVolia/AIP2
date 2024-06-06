import math
import numpy as np
import matplotlib.pyplot as plt


def plot_data_sample(data, num_images=16):
    x, y = data

    # Choose random images from the test set for visualization
    random_indices = np.random.choice(len(x), size=num_images, replace=False)
    images = x[random_indices]
    true_labels = np.argmax(y[random_indices], axis=1)

    # Define class names for CIFAR-10 dataset
    # fmt: off
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    # fmt: on

    # Visualize images with true and predicted labels
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, math.ceil(num_images / 2), i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {class_names[true_labels[i]]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_loss_and_accuracy(history):
    # Extract loss and accuracy metrics from the history object
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(1, len(train_loss) + 1)

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, "b", label="Training Accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_results(model, test_dataset):
    x_test, y_test = test_dataset

    # Choose random images from the test set for visualization
    num_images = 10
    random_indices = np.random.choice(len(x_test), size=num_images, replace=False)
    images = x_test[random_indices]
    true_labels = np.argmax(y_test[random_indices], axis=1)

    # Get model predictions
    predicted_labels = np.argmax(model.predict(images), axis=1)

    # fmt: off
    # Define class names for CIFAR-10 dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    # fmt: on

    # Visualize images with true and predicted labels
    plt.figure(figsize=(9, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(
            f"True: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_labels[i]]}",
            fontsize=10,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()
