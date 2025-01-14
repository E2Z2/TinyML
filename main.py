import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt




class TinyNeuralNetwork(object):

    #initializing the parameters for the TinyNN
    def __init__(self, num_input: int, num_hidden: int, num_output: int):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.a1 = 0
        self.z1 = 0
        self.grad = {}
        self.loss_all = []
        self.acc_all = []
        self.acc_all_test = []
        self.false_class = []
        self.theta = {}

        #starting with random weigths
        self.theta['W0'] = np.random.normal(0, 0.05, (self.num_hidden, self.num_input))
        self.theta['W1'] = np.random.normal(0, 0.05, (self.num_output, self.num_hidden))
        self.theta['b0'] = np.random.normal(0, 0.05, (self.num_hidden)).reshape(-1, 1)
        self.theta['b1'] = np.random.normal(0, 0.05, (self.num_output)).reshape(-1, 1)
        

    #Neuron Activiation function in our Case SiLU
    def activation(self, z):
        return (z) / (1 + np.exp(-z))

    #Derivative of the Activation function
    def d_activation(self, z):
        return (z * np.exp(-z) + np.exp(-z) + 1)/((1 + np.exp(-z)) ** 2)

    #the softmax function so that the output is between 0 and 1 (the sum of all outputs)
    def softmax(self, z):
        return np.exp(z) / np.exp(z).sum(axis=0)

    #cross entropy loss to calculate the loss of the current NN for the adjustment algorithm
    def cross_entropy_loss(self, y_tilde, y_true):
        return -np.sum(y_true * np.log(y_tilde + 1e-12), axis = 0) 

    #letting input data pass forward through the neural network and giving back the prediction
    def forward(self, x):

        z1 = np.einsum('ij,jk->ik', self.theta["W0"], x) + self.theta["b0"]
        a1 = self.activation(z1)

        self.z1 = z1
        self.a1 = a1

        z2 = np.einsum('ij,jk->ik', self.theta["W1"], a1) + self.theta["b1"]
        y_tilde = self.softmax(z2)


        return y_tilde
    
    #backpropagating with the calculation of the gradient values for the adjustment algorithm
    def backward(self, x, y_one_hot, y_tilde):

        dl_dz2 = (y_tilde - y_one_hot) / y_one_hot.shape[1]
        dl_db1 = np.sum(dl_dz2, axis = 1, keepdims = True)
        dl_dW1 = np.einsum('ik,jk->ij', dl_dz2, self.a1)

        dl_da1 = np.einsum('ij,jk->ik', self.theta['W1'].transpose(), dl_dz2)

        dl_buf = dl_da1 * self.d_activation(self.z1)

        dl_db0 = np.sum(dl_buf, axis=1, keepdims=True)
        dl_dW0 = np.einsum('ik,jk->ij', dl_buf, x)

        
        self.grad = {"W0" : dl_dW0, "b0" : dl_db0, "W1" : dl_dW1, "b1" : dl_db1}

    #adjustment algorithm is called
    def gradient(self, x, y_one_hot, step_size, iterations, x_test, y_one_hot_test):
        self.nag(x, y_one_hot, step_size, iterations, x_test, y_one_hot_test)

    #calculates the accuracy of the predictions, optional: if the erros are saved
    def calc_acc(self, y_tilde, y_one_hot, save = False):
        acc = 0
        y_tilde_t = y_tilde.transpose()
        y_one_hot_t = y_one_hot.transpose()
        for i in range(y_tilde.shape[1]):
            if(y_tilde_t[i].argmax() == y_one_hot_t[i].argmax()):
                acc += 1
            elif (save):
                self.false_class.append((i, y_tilde_t[i].argmax()))
        return (acc / y_tilde.shape[1])

    #helper function to make deep copies of the data dictionaries
    def deep_copy_dict(self, original_dict):
        copied_dict = {}
        for key, value in original_dict.items():
            copied_dict[key] = np.copy(value)
        return copied_dict

    #the adjustment algorithm for the weights and biasis
    def nag(self, x, y_one_hot, step_size, iterations, x_test, y_one_hot_test):
        theta_k = 0
        p_km1 = self.deep_copy_dict(self.theta)
        for k in range(iterations):

            y_tilde_test = self.forward(x_test)

            acc_test = self.calc_acc(y_tilde_test, y_one_hot_test)

            self.acc_all_test.append(acc_test)



            y_tilde = self.forward(x)

            acc = self.calc_acc(y_tilde, y_one_hot)

            self.acc_all.append(acc)

            loss = self.cross_entropy_loss(y_tilde, y_one_hot).mean()

            self.loss_all.append(loss)

            #update
            theta_k1 = (1 + np.sqrt(1 + 4 * (theta_k ** 2))) / 2
            p_k = self.deep_copy_dict(self.theta)
            for key, val in self.theta.items():
                self.theta[key] += ((theta_k - 1)/ theta_k1) * (self.theta[key] - p_km1[key])
            p_km1 = p_k
            self.backward(x, y_one_hot, y_tilde)
            for key, val in self.theta.items():
                self.theta[key] = self.theta[key] - step_size * self.grad[key]
            theta_k = theta_k1

        return 0

    #helper function to export the model as json weights and biasis
    def export_model(self):
        with open(f'model_tiny.json', 'w') as file:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, file)

    #helper function to export the import as json 
    def import_model(self): 
        with open(f'model_tiny.json', 'r') as file: 
            data = json.load(file) 
            self.theta = {key: np.array(value) for key, value in data.items()}

    #gives back the accuracy of the given test sample
    def test_accuracy(self, x_test, y_one_hot_test):
        y_tilde_test = self.forward(x_test)

        acc_test = self.calc_acc(y_tilde_test, y_one_hot_test, True)
        return acc_test
    
    #draws the give number and shows it for presentation
    def draw(self, number):
        array_2d = number.reshape((28, 28))
        plt.imshow(array_2d)
        plt.show()

    


#encode the y_data as one hot vectors
def one_hot_encode(y_data, n):
    labels_one_hot = np.zeros((y_data.shape[0], n))
    labels_one_hot[np.arange(y_data.shape[0]), y_data.astype(np.uint8)] = 1.0
    return labels_one_hot


#trains the neural network on the given training data and exports the model 
def train(data):
    #seed so the results can be comparable
    np.random.seed(10)

    input_dim = 784
    hidden_size = 16
    output_dim = 10


    TinyNL = TinyNeuralNetwork(input_dim, hidden_size, output_dim)

    x = data["x_train"].transpose()
    y_one_hot = one_hot_encode(data["y_train"], 10).transpose()

    x_test = data["x_test"].transpose()
    y_one_hot_test = one_hot_encode(data["y_test"], 10).transpose()

    TinyNL.gradient(x, y_one_hot, 0.05, 250, x_test, y_one_hot_test)
    TinyNL.export_model()

#test the given data on either the trained or untrained model
def test(data, load = False):
    net= TinyNeuralNetwork(784, 16, 10)
    x_test = data["x_test"].transpose()
    y_one_hot_test = one_hot_encode(data["y_test"], 10).transpose()
    if (load):
        net.import_model()
    print(net.test_accuracy(x_test, y_one_hot_test))

    
#shows examples of the data where the NN failed
def show_failure(data):
    net= TinyNeuralNetwork(784, 16, 10)
    x_test = data["x_test"].transpose()
    y_one_hot_test = one_hot_encode(data["y_test"], 10).transpose()
    net.import_model()
    net.test_accuracy(x_test, y_one_hot_test)
    c = 0
    for x, y in net.false_class:
        print("Prediction: " + str(y))
        print("Should be: " + str(y_one_hot_test.transpose()[x].argmax()))
        net.draw(x_test.transpose()[x])
        c += 1
        if (c >= 3):
            break

#test the give picture self drawn
def test_picture(x_data):
    net= TinyNeuralNetwork(784, 16, 10)
    net.import_model()
    test = []
    test.append(x_data.flatten())
    test = np.array(test).transpose()
    print(net.forward(test).argmax())
    net.draw(test)

#imports a png and converts the input to binary black, white
def import_image(image_path):
    image = Image.open(image_path).convert('L')  
    if image.size != (28, 28):
        image = image.resize((28, 28))
    image_array = np.array(image)

    return (image_array < 150).astype(np.uint8)

if __name__ == '__main__':

    with np.load('train.npz') as train_data:
        x_train = train_data['array1']
        y_train = train_data['array2']

    with np.load('test.npz') as test_data:
        x_test= test_data['array1']
        y_test = test_data['array2']

    data = {"x_train" : x_train[:15000], "y_train" : y_train[:15000], "x_test" : x_test[:1000], "y_test" : y_test[:1000]}

    parser = argparse.ArgumentParser(description='Argument Option to pass') 

    parser.add_argument('Mode', type=str, help='Either test, train, show')

    arguments = parser.parse_args()

    if arguments.Mode == "train":
        train(data)
    elif arguments.Mode == "test1":
        test(data)
    elif arguments.Mode == "test2":
        test(data, True)
    elif arguments.Mode == "testown":
        test_picture(import_image("1.png"))
    elif arguments.Mode == "failure":
        show_failure(data)
    else:
        print("This mode does not exist!")

    




    
