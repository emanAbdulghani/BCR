import csv
import numpy as np
from keras.datasets import mnist
import tensorflow
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(TrainX, TrainY), (TestX, TestY) = mnist.load_data()


def display_img(mnist_index):
    image = mnist_index
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def image_grid(image, row, col):
    x, y = image.shape
    return (image.reshape(x // row, row, -1, col).swapaxes(1, 2).reshape(-1, row, col))


def centroid(image):
    feature_vector = []
    for grid in image_grid(image, 7, 7):
        Xc = 0
        Yc = 0
        Sum = 0
        for idx, x in np.ndenumerate(grid):
            Sum += x
            Xc += x * idx[0]
            Yc += x * idx[1]

            if Sum != 0:
                feature_vector.append(Xc / Sum)
                feature_vector.append(Yc / Sum)
            else:
                feature_vector.append(0)
                feature_vector.append(0)
    return np.array(feature_vector)


train_features = [centroid(img) for img in TrainX]
train_features = np.array(train_features)
test_features = [centroid(img) for img in TestX]
test_features = np.array(test_features)


def KNN(train_features, test_features, train_labels):
    knn = KNeighborsClassifier(1, metric='euclidean')
    knn.fit(train_features, TrainY)
    prediction = knn.predict(test_features)
    return prediction


Knn_Prediction = KNN(train_features, test_features, TrainY)
print("Accuracy Score =", accuracy_score(TestY, Knn_Prediction) * 100, "%")