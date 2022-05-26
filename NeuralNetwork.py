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
            input_values[i] = train_data[i][0:4]

        for i in range(0, len(input_test_values)):
            input_test_values[i] = test_data[i][0:4]

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
                acc_test = self.accuracy(test_data)
                acc_train = self.accuracy(train_data)

                error_list[1].append(generations)
                error_list[0].append(mse)


                print("generations = " + str(generations) + " ms error = %0.4f " % mse + " accuracy train = %0.4f " % acc_train + " accuracy test = %0.4f " % acc_test)

                if((acc_train > acc_train_prev) and (acc_test >= acc_test_prev)) or ((acc_train >= acc_train_prev) and (acc_test > acc_test_prev)):
                    acc_test_prev = acc_test
                    acc_train_prev = acc_train
                    mse_prev = mse
                    gen_prev = generations
                    backup_network = self
                elif((acc_train == acc_train_prev) and (acc_test == acc_test_prev)):
                    if(mse < mse_prev):
                        acc_test_prev = acc_test
                        acc_train_prev = acc_train
                        mse_prev = mse
                        gen_prev = generations
                        backup_network = self

        finalList = []
        finalList.append(gen_prev)
        finalList.append(mse_prev)
        finalList.append(acc_test_prev)
        finalList.append(acc_train_prev)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        uuid = shortuuid.ShortUUID().random(length=10)
        finalList.append(uuid)
        path = folder_name + "/" + uuid
        os.mkdir(path)

        save_network(backup_network, str(path)+"/best_record_of_network"+".pkl")
        save_network(self, str(path) + "/record_of_network.pkl")
        iov.save_val(ioval_train, str(path) + "/ioval_train_data" + ".pkl")
        iov.save_val(ioval_test, str(path) + "/ioval_test_data" + ".pkl")

        with open(path+"/error_gen_list.txt", 'a') as f:
            for i in range(0, len(error_list[0])):
                f.write("Gen: " + str(error_list[1][i]) + " | Error: " + str(error_list[0][i]) + '\n')
            f.close()

        self.draw_plot(error_list, path)
        self.confusion_matrix(train_data, path, "train")
        self.confusion_matrix(test_data, path, "test")



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

    def accuracy(self, tdata):
        num_correct = 0
        num_wrong = 0
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)

        for i in range(len(tdata)):
            input_values = tdata[i]


            if int(tdata[i][4]) == 1:
                output_values = np.array([0, 1, 0], dtype = np.float32)
            elif int(tdata[i][4]) == 0:
                output_values = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values = np.array([0, 0, 1], dtype = np.float32)

            output_after_train = self.counting_outputs(input_values)
            max_index = np.argmax(output_after_train)
            if abs(output_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct) / (num_correct + num_wrong)

    def squared_error(self, tdata):  # on train or test data matrix

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

    def draw_matrix(self, matrix, f):
        for i in range(0, 3):
            for j in range(0, 3):
                f.write(str(matrix[i][j]) + " ")
            f.write('\n')

        f.write('\n')
        f.write("Klasa I" + "\n")
        TP = matrix[0][0]
        TN = matrix[1][1] + matrix[2][2] + matrix[1][2] + matrix[2][1]
        FP = matrix[1][0] + matrix[2][0]
        FN = matrix[0][1] + matrix[0][2]

        f.write("TP: " + str(TP) + " | TN: " + str(TN) + " | FP: " + str(FP) + " | FN: " + str(FN) + "\n")

        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN) * 1.0
        specificity = TN * 1.0 / (TN + FP) * 1.0

        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("Specificity: " + str(specificity) + "\n")
        f.write("f-measure: " + str(2 * ((precision * recall) / precision + recall)) + "\n")

        f.write('\n')
        f.write("Klasa II" + "\n")
        TP = matrix[1][1]
        TN = matrix[0][0] + matrix[0][2] + matrix[2][0] + matrix[2][2]
        FP = matrix[0][1] + matrix[2][1]
        FN = matrix[1][0] + matrix[1][2]

        f.write("TP: " + str(TP) + " | TN: " + str(TN) + " | FP: " + str(FP) + " | FN: " + str(FN) + "\n")

        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN) * 1.0
        specificity = TN * 1.0 / (TN + FP) * 1.0

        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("Specificity: " + str(specificity) + "\n")
        f.write("f-measure: " + str(2 * ((precision * recall) / precision + recall)) + "\n")

        f.write('\n')
        f.write("Klasa III" + "\n")
        TP = matrix[2][2]
        TN = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
        FP = matrix[0][2] + matrix[1][2]
        FN = matrix[2][0] + matrix[2][1]

        f.write("TP: " + str(TP) + " | TN: " + str(TN) + " | FP: " + str(FP) + " | FN: " + str(FN) + "\n")

        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN) * 1.0
        specificity = TN * 1.0 / (TN + FP) * 1.0

        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("Specificity: " + str(specificity) + "\n")
        f.write("f-measure: " + str(2 * ((precision * recall) / precision + recall)))

    def confusion_matrix(self, tdata, path, name):
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

            output_after_train = self.counting_outputs(input_values)
            max_index = np.argmax(output_after_train)
            if int(max_index) == int(tdata[i][4]):
                matrix[max_index][max_index] += 1
            else:
                matrix[int(tdata[i][4])][int(max_index)] += 1

        with open(path+"/confusion_matrix_"+name+".txt", 'a') as f:
            self.draw_matrix(matrix, f)
            f.close()

    def think(self, tdata, path):
        input_values = np.zeros(shape=[self.input.amount], dtype=np.float32)
        output_values = np.zeros(shape=[self.output.amount], dtype=np.float32)

        matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        acc = None
        err = None

        for i in range(len(tdata)):
            input_values = tdata[i][0:4]


            if int(tdata[i][4]) == 1:
                output_values = np.array([0, 1, 0], dtype = np.float32)
            elif int(tdata[i][4]) == 0:
                output_values = np.array([1, 0, 0], dtype = np.float32)
            else:
                output_values = np.array([0, 0, 1], dtype = np.float32)

            output_after_train = self.counting_outputs(input_values)


            max_index = np.argmax(output_after_train)
            if int(max_index) == int(tdata[i][4]):
                matrix[max_index][max_index] += 1
            else:
                matrix[int(tdata[i][4])][int(max_index)] += 1

            acc = self.accuracy(tdata)
            err = self.squared_error(tdata)

            with open(path, 'a') as f:
                f.write("Dla iteracji: [" + str(i + 1) + "/" + str(len(tdata)) + "]" + '\n')
                f.write("Dane wejściowe: " + str(input_values) + " | Dane wyjsciowe: " + str(output_values) + "\n")
                f.write("Wartości otrzymane z sieci: " + str(output_after_train) + '\n')
                f.write("\n")
                f.write("WARSTWA UKRYTA\n")
                f.write("Wartości wyjściowe w warstwie ukrytej: " + str(self.hidden.neurons) + '\n')
                f.write("Błędy popełnione na wyjściu w warstwie ukrytej: " + str(self.hidden.error) + '\n')
                f.write("Wartości wyjściowe w warstwie ukrytej: " + str(self.hidden.error) + '\n')
                f.write("Wagi neuronów pomiędzy warstwą wejściową a warstwą ukrytą:\n")
                for i in range(0, len(self.in_hid_weights)):
                    f.write(str(self.in_hid_weights[i]) + "\n")

                f.write("\n")
                f.write("WARSTWA WYJSCIOWA\n")
                f.write("Wartości wyjściowe w warstwie wyjściowej: " + str(self.output.neurons) + '\n')
                f.write("Błędy popełnione na wyjściu w warstwie wyjściowej: " + str(self.output.error) + '\n')
                f.write("Wartości wyjściowe w warstwie wyjściowej: " + str(self.output.error) + '\n')
                f.write("Wagi neuronów pomiędzy warstwą ukrytą a warstwą wyjściową:\n")
                for i in range(0, len(self.hid_out_weights)):
                    f.write(str(self.hid_out_weights[i]) + "\n")
                f.write("\n")
                f.write("----------------------------------------------------------------------------------\n")
                f.write("\n")
                f.close()

        with open(path, 'a') as f:
            f.write("PODSUMOWANIE:\n")
            f.write("Error: " + str(err) + '\n')
            f.write("Accuracy: " + str(acc) + '\n')
            f.write("\n")
            self.draw_matrix(matrix, f)



