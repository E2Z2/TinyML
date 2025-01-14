from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def flatten_data(data):
    flatten = []
    for x in data:
        flatten.append(x.flatten())

    return np.array(flatten)


(X_train, y_train), (X_test, y_test) = mnist.load_data()


#flattening it for Neural Network

X_train_flatten = flatten_data(X_train)
X_test_flatten = flatten_data(X_test)

#Making the data binary 
X_train_flatten[X_train_flatten != 0] = 1
X_test_flatten[X_test_flatten != 0] = 1

np.savez('train.npz', array1=X_train_flatten, array2=y_train)
np.savez('test.npz', array1=X_test_flatten, array2=y_test)
