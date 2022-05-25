import numpy as np
import random
import WeightsNeurons as w
import shortuuid
import IOValues as iov
import matplotlib.pyplot as plt
import os


class NeuralNetwork:

    def __init__(self, input_layer, hidden_layer, output_layer, seed):
        self.input = input_layer
        self.hidden = hidden_layer
        self.output = output_layer

        self.in_hid_weights = 2 * np.random.random((self.input.amount, self.hidden.amount)) - 1

        self.hid_out_weights = 2 * np.random.random((self.hidden.amount, self.output.amount)) - 1

        self.hid_out_grads = 2 * np.random.random((self.hidden.amount, self.output.amount)) - 1

        self.in_hid_grads = 2 * np.random.random((self.input.amount, self.hidden.amount)) - 1

        self.ih_prev_weights_delta = np.zeros(shape=[self.input.amount, self.hidden.amount], dtype=np.float32)
        self.ho_prev_weights_delta = np.zeros(shape=[self.hidden.amount, self.output.amount], dtype=np.float32)

        self.rnd = random.Random(seed)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def drawPlot(self, error_values, path):
        plt.plot(error_values[1], error_values[0])
        plt.xlabel("Liczba epok")
        plt.ylabel("Błąd średniokwadratowy")
        plt.savefig(path + "/weights_chart.png")

    def computeOutputs(self, input_values):

        hidden_neurons_value = np.zeros(shape=[self.hidden.amount], dtype=np.float32)
        output_neurons_value = np.zeros(shape=[self.output.amount], dtype=np.float32)

        self.input.neurons = input_values

        for i in range(self.hidden.amount):
            for j in range(self.input.amount):
                hidden_neurons_value[i] += (self.input.neurons[j] * self.in_hid_weights[j, i]) + self.hidden.biases[i]

        self.hidden.neurons = self.sigmoid(hidden_neurons_value)

        for i in range(self.output.amount):
            for j in range(self.hidden.amount):
                output_neurons_value[i] += (self.hidden.neurons[j] * self.hid_out_weights[j, i]) + self.output.biases[i]

        self.output.neurons = self.sigmoid(output_neurons_value)

        return self.output.neurons

    def train(self, train_data,test_data, max_generations, learn_rate, momentum, folder_name):

        mse_prev = 10
        acc_test_prev = 0
        acc_train_prev = 0
        gen = 0

        save_weights = None

        error_list = [[], []]

        generations = 0
        input_values = np.zeros(shape=[len(train_data), self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[len(train_data), self.output.amount], dtype=np.float32)

        input_test_values = np.zeros(shape=[len(test_data), self.input.amount], dtype=np.float32)
        output_test_values = np.zeros(shape=[len(test_data), self.output.amount], dtype=np.float32)

        numTrainItems = len(train_data)

        input_values = train_data
        input_test_values = test_data

        for i in range(len(train_data)):
            if int(train_data[i][4]) == 1:
                output_values[i] = np.array([0, 1, 0], dtype = np.float32)
            elif int(train_data[i][4]) == 0:
                output_values[i] = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values[i] = np.array([0, 0, 1], dtype = np.float32)

        for i in range(len(test_data)):
            if int(test_data[i][4]) == 1:
                output_test_values[i] = np.array([0, 1, 0], dtype = np.float32)
            elif int(test_data[i][4]) == 0:
                output_test_values[i] = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_test_values[i] = np.array([0, 0, 1], dtype = np.float32)

        ioval_train = iov.IOValues(input_values, output_values)
        ioval_test = iov.IOValues(input_test_values, output_test_values)

        while generations < max_generations:
            for index in range(numTrainItems):

                self.computeOutputs(input_values[index])

                derivative = self.sigmoid_derivative(self.output.neurons)
                self.output.error = derivative * (output_values[index] - self.output.neurons)

                for i in range(self.hidden.amount):
                    for j in range(self.output.amount):
                        self.hid_out_grads[i, j] = self.output.error[j] * self.hidden.neurons[i]

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

                self.hidden.biases_grads = self.hidden.error * 1.0

                delta = learn_rate * self.in_hid_grads
                self.in_hid_weights += delta + (momentum * self.ih_prev_weights_delta)
                self.ih_prev_weights_delta = delta

                delta = learn_rate * self.hidden.biases_grads * 1.0
                self.hidden.biases += delta + (momentum * self.hidden.prev_biases_delta)
                self.hidden.prev_biases_delta = delta

                delta = learn_rate * self.hid_out_grads
                self.hid_out_weights += delta + (momentum * self.ho_prev_weights_delta)
                self.ho_prev_weights_delta = delta

                delta = learn_rate * self.output.biases_grads
                self.output.biases += delta + (momentum * self.output.prev_biases_delta)
                self.output.prev_biases_delta = delta

            generations += 1

            if generations % 20 == 0:
                mse = self.meanSquaredError(test_data)
                acc_test = self.accuracy(test_data)
                acc_train = self.accuracy(train_data)

                error_list[1].append(generations)
                error_list[0].append(mse)


                print("generations = " + str(generations) + " ms error = %0.4f " % mse + " accuracy train = %0.4f " % acc_train + " accuracy test = %0.4f " % acc_test)

                if((acc_train > acc_train_prev) and (acc_test >= acc_test_prev)) or ((acc_train >= acc_train_prev) and (acc_test > acc_test_prev)):
                    acc_test_prev = acc_test
                    acc_train_prev = acc_train
                    mse_prev = mse
                    gen = generations
                    save_weights = w.weightsNeurons(self.in_hid_weights,
                                                    self.hid_out_weights,
                                                    self.hid_out_grads,
                                                    self.in_hid_grads,
                                                    self.ih_prev_weights_delta,
                                                    self.ho_prev_weights_delta)
                elif((acc_train == acc_train_prev) and (acc_test == acc_test_prev)):
                    if(mse < mse_prev):
                        acc_test_prev = acc_test
                        acc_train_prev = acc_train
                        mse_prev = mse
                        gen = generations
                        save_weights = w.weightsNeurons(self.in_hid_weights,
                                                        self.hid_out_weights,
                                                        self.hid_out_grads,
                                                        self.in_hid_grads,
                                                        self.ih_prev_weights_delta,
                                                        self.ho_prev_weights_delta)






        # end while
        finalList = []
        finalList.append(gen)
        finalList.append(mse_prev)
        finalList.append(acc_test_prev)
        finalList.append(acc_train_prev)
        acc_percentage = ((acc_train_prev+acc_test_prev)/2)*100
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        uuid = shortuuid.ShortUUID().random(length=10) + "_" + str(round(acc_percentage, 2)).replace('.', '-')
        finalList.append(uuid)
        path = folder_name + "/" + uuid
        os.mkdir(path)
        w.save_weights(save_weights, str(path)+"/weights_"+str(round(acc_percentage, 2)).replace('.', ',')+".pkl")
        iov.save_val(ioval_train, str(path) + "/ioval_train_data" + ".pkl")
        iov.save_val(ioval_test, str(path) + "/ioval_test_data" + ".pkl")

        self.drawPlot(error_list, path)
        self.last_table(train_data)

        return finalList

    def accuracy(self, tdata):  # train or test data matrix
        num_correct = 0
        num_wrong = 0
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item # peel off input values from curr data row
            input_values = tdata[i]


            if int(tdata[i][4]) == 1:
                output_values = np.array([0, 1, 0], dtype = np.float32)
            elif int(tdata[i][4]) == 0:
                output_values = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values = np.array([0, 0, 1], dtype = np.float32)

            output_after_train = self.computeOutputs(input_values)  # computed output values)
            max_index = np.argmax(output_after_train)  # index of largest output value
            if abs(output_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1
                #print("Otrzymany wynik: " + str(output_after_train) + " poprawny wynik: " + str(output_values))

        return (num_correct * 1.0) / (num_correct + num_wrong)

    def meanSquaredError(self, tdata):  # on train or test data matrix

        sumSquaredError = 0.0
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item # peel off input values from curr data row
            input_values = tdata[i]

            if int(tdata[i][4]) == 1:
                output_values = np.array([0, 1, 0], dtype = np.float32)
            elif int(tdata[i][4]) == 0:
                output_values = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values = np.array([0, 0, 1], dtype = np.float32)

            y_values = self.computeOutputs(input_values)  # computed output values

            for j in range(self.output.amount):
                err = output_values[j] - y_values[j]
                sumSquaredError += err * err

        return sumSquaredError / len(tdata)

    def last_table(self, tdata):
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)


        matrix = [[0,0,0],[0,0,0],[0,0,0]]


        for i in range(len(tdata)):
            input_values = tdata[i]


            if int(tdata[i][4]) == 1:
                output_values = np.array([0, 1, 0], dtype = np.float32)
            elif int(tdata[i][4]) == 0:
                output_values = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values = np.array([0, 0, 1], dtype = np.float32)

            output_after_train = self.computeOutputs(input_values)
            max_index = np.argmax(output_after_train)
            if int(max_index) == int(tdata[i][4]):
                matrix[max_index][max_index] += 1
            else:
                matrix[int(tdata[i][4])][int(max_index)] += 1

        for i in range(0, 3):
            for j in range(0, 3):
                print(str(matrix[i][j]) + " ", end = '')
            print("")


        print("Klasa I")
        print("TP: " + str(matrix[0][0]))
        print("TN: " + str(matrix[1][1] + matrix[2][2] + matrix[1][2] + matrix[2][1]))
        print("FP: " + str(matrix[1][0] + matrix[2][0]))
        print("FN: " + str(matrix[0][1] + matrix[0][2]))




