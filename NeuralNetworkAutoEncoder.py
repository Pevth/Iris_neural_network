import numpy as np
import dill
import shortuuid
import IOValues as iov
import matplotlib.pyplot as plt
import os

def save_network(object, filename):
    dill.dump(object, file = open(filename, "wb"))

def load_network(filename):
    return dill.load(open(filename, "rb"))

class NeuralNetwork:

    def __init__(self, input_layer, hidden_layer, output_layer, bias):
        self.input = input_layer
        self.hidden = hidden_layer
        self.output = output_layer

        self.in_hid_weights = 2 * np.random.random((self.input.amount, self.hidden.amount)) - 1

        self.hid_out_weights = 2 * np.random.random((self.hidden.amount, self.output.amount)) - 1

        self.hid_out_grads = 2 * np.random.random((self.hidden.amount, self.output.amount)) - 1

        self.in_hid_grads = 2 * np.random.random((self.input.amount, self.hidden.amount)) - 1

        self.ih_prev_weights_delta = np.zeros(shape=[self.input.amount, self.hidden.amount], dtype=np.float32)
        self.ho_prev_weights_delta = np.zeros(shape=[self.hidden.amount, self.output.amount], dtype=np.float32)

        self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, train_data, test_data, max_generations, learn_rate, momentum, folder_name, step_generations):

        ioval_train = iov.IOValues(train_data)
        ioval_test = iov.IOValues(test_data)

        mse_prev = 10
        acc_test_prev = 0
        acc_train_prev = 0
        gen_prev = 0

        save_weights = None
        backup_network = None

        error_list = [[], []]

        generations = 0
        input_values = np.zeros(shape=[len(train_data), self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[len(train_data), self.output.amount], dtype=np.float32)

        input_test_values = np.zeros(shape=[len(test_data), self.input.amount], dtype=np.float32)
        output_test_values = np.zeros(shape=[len(test_data), self.output.amount], dtype=np.float32)

        numTrainItems = len(train_data)

        for i in range(0, len(input_values)):
            input_values[i] = train_data[i]
            output_values[i] = train_data[i]

        for i in range(0, len(input_test_values)):
            input_test_values[i] = test_data[i]
            output_test_values[i] = test_data[i]


        while generations < max_generations:
            for index in range(numTrainItems):

                self.counting_outputs(input_values[index])

                derivative = self.sigmoid_derivative(self.output.neurons)
                self.output.error = derivative * (output_values[index] - self.output.neurons)

                for i in range(self.hidden.amount):
                    for j in range(self.output.amount):
                        self.hid_out_grads[i, j] = self.output.error[j] * self.hidden.neurons[i]

                if self.bias:
                    self.output.biases_grads = self.output.error * 1.0

                for j in range(self.hidden.amount):
                    sum = 0.0
                    for k in range(self.output.amount):
                        sum += self.output.error[k] * self.hid_out_weights[j, k]
                    derivative = self.sigmoid_derivative(self.hidden.neurons[j])
                    self.hidden.error[j] = derivative * sum

                for i in range(self.input.amount):
                    for j in range(self.hidden.amount):
                        self.in_hid_grads[i, j] = self.hidden.error[j] * self.input.neurons[i]

                if self.bias:
                    self.hidden.biases_grads = self.hidden.error * 1.0

                delta = learn_rate * self.in_hid_grads
                self.in_hid_weights += delta + (momentum * self.ih_prev_weights_delta)
                self.ih_prev_weights_delta = delta

                if self.bias:
                    delta = learn_rate * self.hidden.biases_grads * 1.0
                    self.hidden.biases += delta + (momentum * self.hidden.prev_biases_delta)
                    self.hidden.prev_biases_delta = delta

                delta = learn_rate * self.hid_out_grads
                self.hid_out_weights += delta + (momentum * self.ho_prev_weights_delta)
                self.ho_prev_weights_delta = delta

                if self.bias:
                    delta = learn_rate * self.output.biases_grads
                    self.output.biases += delta + (momentum * self.output.prev_biases_delta)
                    self.output.prev_biases_delta = delta

            generations += 1

            if generations % step_generations == 0:
                mse = self.squared_error(train_data)

                error_list[1].append(generations)
                error_list[0].append(mse)


                print("generations = " + str(generations) + " ms error = %0.4f " % mse)

                if(mse < mse_prev):
                    mse_prev = mse
                    gen_prev = generations
                    backup_network = self

        finalList = []
        finalList.append(gen_prev)
        finalList.append(mse_prev)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        uuid = shortuuid.ShortUUID().random(length=10)
        finalList.append(uuid)
        path = folder_name + "/" + uuid
        os.mkdir(path)

        save_network(backup_network, str(path)+"/best_record_of_network"+".pkl")
        save_network(self, str(path) + "/record_of_network.pkl")

        with open(path+"/error_gen_list.txt", 'a') as f:
            for i in range(0, len(error_list[0])):
                f.write("Gen: " + str(error_list[1][i]) + " | Error: " + str(error_list[0][i]) + '\n')
            f.close()

        self.draw_plot(error_list, path)



        return finalList

    def counting_outputs(self, input_values):

        hidden_neurons_value = np.zeros(shape=[self.hidden.amount], dtype=np.float32)
        output_neurons_value = np.zeros(shape=[self.output.amount], dtype=np.float32)

        self.input.neurons = input_values

        for i in range(self.hidden.amount):
            for j in range(self.input.amount):
                if self.bias:
                    hidden_neurons_value[i] += (self.input.neurons[j] * self.in_hid_weights[j, i]) + self.hidden.biases[i]
                else:
                    hidden_neurons_value[i] += (self.input.neurons[j] * self.in_hid_weights[j, i])

        self.hidden.neurons = self.sigmoid(hidden_neurons_value)

        for i in range(self.output.amount):
            for j in range(self.hidden.amount):
                if self.bias:
                    output_neurons_value[i] += (self.hidden.neurons[j] * self.hid_out_weights[j, i]) + self.output.biases[i]
                else:
                    output_neurons_value[i] += (self.hidden.neurons[j] * self.hid_out_weights[j, i])

        self.output.neurons = self.sigmoid(output_neurons_value)

        return self.output.neurons

    def squared_error(self, tdata):  # on train or test data matrix

        sumSquaredError = 0.0
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item # peel off input values from curr data row
            input_values = tdata[i]
            output_values = tdata[i]

            y_values = self.counting_outputs(input_values)  # computed output values

            for j in range(self.output.amount):
                err = output_values[j] - y_values[j]
                sumSquaredError += err * err

        return sumSquaredError / len(tdata)

    def draw_plot(self, error_values, path):
        plt.plot(error_values[1], error_values[0])
        plt.xlabel("Liczba epok")
        plt.ylabel("Błąd średniokwadratowy")
        plt.savefig(path + "/weights_chart.png")





