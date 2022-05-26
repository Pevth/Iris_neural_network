import os

import shortuuid
import FileGenerator as fG
import NeuronLayer as nL
import NeuralNetwork as nN
import IOValues as iov
from multiprocessing import Process, Manager
from time import perf_counter

def replace_dot(s):
    return str(s).replace('.', ',')

def ProcessTrain(iterations, n_network, max_generations, learn_rate, momentum, arr, folder_name, uuid, step_generations):
    for i in range(0, iterations):
        trainDataMatrix, testDataMatrix = fG.randomDataSets("fullData.csv")
        list = n_network.train(trainDataMatrix, testDataMatrix, max_generations, learn_rate, momentum, folder_name,uuid, step_generations)
        arr.append(list)
        nN.save_network(n_network, folder_name + "/" + uuid + "/record_of_network.pkl")
        n_network.think(testDataMatrix, folder_name + "/" + uuid + "/train_result.txt")

def main():
    print("\nRozpoczęcie dzialania sieci\n")

    seed = 3

    input = nL.NeuronLayer(4)
    hidden = nL.NeuronLayer(6)
    output = nL.NeuronLayer(3)

    print("Utworzenie sieci zlożonej z: %d-%d-%d neuronów" % (input.amount, hidden.amount, output.amount))

    max_generations = 1000
    learn_rate = 0.05
    momentum = 0.7
    bias = True

    print("\nMaksymalna ilość generacji = " + str(max_generations))
    print("Learning rate: = %0.3f " % learn_rate)
    print("Momentum: = %0.3f " % momentum)

    n_network = nN.NeuralNetwork(input, hidden, output, bias)

    process = list()
    fullList = []
    start_time = perf_counter()
    folder_name = str(input.amount) + "-" + str(hidden.amount) + "-" \
                  + str(output.amount) + "_M" + str(momentum) \
                  + "_LR" + str(learn_rate) + "_G" + str(max_generations) + "_B-"+str(bias)[0]

    with Manager() as manager:
        L = manager.list()
        for i in range(0, 1):
            uuid = shortuuid.ShortUUID().random(length=10)
            x = Process(target=ProcessTrain,
                        args=(n_network, max_generations, learn_rate, momentum, L, folder_name, uuid))
            process.append(x)
            x.start()

        for p in process:
            p.join()
        L = list(L)

    end_time = perf_counter()

    f = open(folder_name + "/all_results.txt", "a")
    for i in range(0, len(L)):
        f.write(str(L[i][4]) + ";" + str(L[i][0]) + ";" + str(replace_dot(L[i][1]))
                + ";" + str(replace_dot(L[i][2])) + ";" + str(replace_dot(L[i][3])) + '\n')

    trainDataMatrix, testDataMatrix = fG.randomDataSets("fullData.csv")
    f.close

    print(f'Program potrzebował {end_time - start_time: 0.2f} sekund na nauczenie się.')

def learn_main():
    input_layer = nL.NeuronLayer(4)
    x = int(input("Podaj liczbe neuronów ukrytych: "))
    if x <= 0:
        print("Podales zla wartosc, ustawiam liczbe na 6")
        x = 6
    hidden_layer = nL.NeuronLayer(x)
    output_layer = nL.NeuronLayer(3)

    print("Utworzenie sieci zlożonej z: %d-%d-%d neuronów" % (input_layer.amount, hidden_layer.amount, output_layer.amount))

    max_generations = int(input("Podaj liczbe generacji: "))

    step_generations = int(input("Podaj co ile generacji sieć będzie sprawdzać wyniki: "))
    if(step_generations > max_generations):
        print("podano złą wartość, ustawiam krok kontroli co 50 generacji")
        step_generations = 50

    learn_rate = float(input("Podaj learning rate [0-1]: "))
    if learn_rate >= 1 and learn_rate <= 0:
        print("podaleś złą wartość, ustawiam lR na 0.05")
        learn_rate = 0.05
    momentum = float(input("Podaj momentum [0-1]: "))
    if learn_rate >= 1 and learn_rate <= 0:
        print("podaleś za dużą wartość, ustawiam momentum na 0.7")
        momentum = 0.7

    x = input("Czy uwzględnić bias? [t/n]: ")

    if(x == 't' or x == 'T'):
        bias = True
    elif(x == 'N' or x == 'n'):
        bias = False
    else:
        print("podaleś złą wartość, ustawiam bias na true")
        bias = True

    n_network = nN.NeuralNetwork(input_layer, hidden_layer, output_layer, bias)
    print("Poprawnie utworzono sieć")

    process = list()

    process_iteration = int(input("Ile chcesz utworzyć procesów? [1-5]: "))
    if process_iteration > 5 or process_iteration < 1:
        print("podano złą wartość, ustawiam wartość na 1")
        process_iteration = 1

    train_iteration = int(input("Ile chcesz utworzyć powtórzeń uczenia? [podaj liczbe]: "))
    if train_iteration < 1:
        print("podano złą wartość, ustawiam wartość na 1")
        train_iteration = 1

    print("Rozpoczynam liczenie działania programu")
    start_time = perf_counter()

    folder_name = str(input_layer.amount) + "-" + str(hidden_layer.amount) + "-" \
                  + str(output_layer.amount) + "_M" + str(momentum) \
                  + "_LR" + str(learn_rate) + "_G" + str(max_generations) + "_B-" + str(bias)[0]

    with Manager() as manager:
        L = manager.list()
        for i in range(0, process_iteration):
            uuid = shortuuid.ShortUUID().random(length=10)
            x = Process(target=ProcessTrain,
                        args=(train_iteration, n_network, max_generations, learn_rate, momentum, L, folder_name, uuid, step_generations))
            process.append(x)
            x.start()

        for p in process:
            p.join()
        L = list(L)

    end_time = perf_counter()

    f = open(folder_name + "/all_results.txt", "a")
    for i in range(0, len(L)):
        f.write(str(L[i][4]) + ";" + str(L[i][0]) + ";" + str(replace_dot(L[i][1]))
                + ";" + str(replace_dot(L[i][2])) + ";" + str(replace_dot(L[i][3])) + '\n')

    f.close

    print(f'Program po {end_time - start_time: 0.2f} sekundach zakończył uczenie.')

def check_main():

    while(True):
        data_path = input("Podaj sciezke do danych: ")
        if os.path.exists(data_path):
            print("Prawidłowo podany plik")
            break
        else:
            print("Plik nie istnieje, bądź została podana zła ścieżka")

    while(True):
        network_path = input("Podaj sciezke do sieci: ")
        if os.path.exists(network_path):
            print("Prawidłowo podany plik")
            break
        else:
            print("Plik nie istnieje, bądź została podana zła ścieżka")

    n_network = nN.load_network(network_path)
    data = iov.load_val(data_path)

    network_uuid = network_path.split("/")
    think_filename = network_uuid[1] + "_think_result.txt"
    think_path = "think_results/" + think_filename

    if not os.path.exists("think_results"):
        os.mkdir("think_results")

    n_network.think(data.values, think_path)
    print("Zapisano wyniki dla sieci: " + network_uuid[1] + "w folderze [think_results]")

if __name__ == "__main__":
    x = int(input("Chcesz trenować czy testować sieć [1/0]: "))

    if x == 1:
        learn_main()
    elif x == 0:
        check_main()
    else:
        print("Podano złą wartość program kończy działanie")



