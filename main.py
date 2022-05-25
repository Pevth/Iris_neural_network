import multiprocessing

import FileGenerator as fG
import NeuronLayer as nL
import NeuralNetwork as nN
from multiprocessing import Process, Manager
from time import perf_counter

def ProcessTrain(nn, maxEpochs, learnRate, momentum, arr, folder_name):
    for i in range(0, 1):
        print(i)
        trainDataMatrix, testDataMatrix = fG.randomDataSets("fullData.csv")
        list = nn.train(trainDataMatrix, testDataMatrix, maxEpochs, learnRate, momentum, folder_name)
        arr.append(list)

if __name__ == "__main__":

    print("\nRozpoczęcie dzialania sieci\n")

    seed = 3

    input = nL.NeuronLayer(4)
    hidden = nL.NeuronLayer(6)
    output = nL.NeuronLayer(3)

    print("Utworzenie sieci zlożonej z: %d-%d-%d neuronów" % (input.amount, hidden.amount, output.amount))

    maxEpochs = 2000
    learnRate = 0.01
    momentum = 0.9

    print("\nMaksymalna ilość generacji = " + str(maxEpochs))
    print("Learning rate: = %0.3f " % learnRate)
    print("Momentum: = %0.3f " % momentum)

    nn = nN.NeuralNetwork(input, hidden, output, seed)

    process = list()
    fullList = []
    start_time = perf_counter()
    folder_name = str(input.amount) + "-" + str(hidden.amount) + "-" + str(output.amount) + "_M"+str(momentum).replace('.', ',') + "_LR" + str(learnRate).replace('.', ',') + "_G"+str(maxEpochs)

    with Manager() as manager:
        L = manager.list()
        for i in range(0, 1):
            q = multiprocessing.Queue()
            x = Process(target = ProcessTrain, args=(nn, maxEpochs, learnRate, momentum,L,folder_name))
            process.append(x)
            x.start()

        for p in process:
            p.join()
        L = list(L)


    end_time = perf_counter()

    count_err=0
    count_train=0
    count_test=0

    print(L)

    f = open(folder_name + "/all_results.txt", "a")
    for i in range(0, len(L)):
        f.write(str(L[i][4])+";"+str(L[i][0]) + ";" + str(L[i][1]).replace('.', ',') + ";" + str(L[i][2]).replace('.', ',') + ";" + str(L[i][3]).replace('.', ',') + '\n')
    f.close

    for i in range(0, len(L)):
        if L[i][1] < 0.05:
            count_err += 1
        if L[i][2] >= 0.95:
            count_train += 1
        if L[i][3] >= 0.95:
            count_test += 1

    print("ERROR UNDER 0.05 ["+str(count_err)+"/500] ACC TRAIN >= 0.95 ["+str(count_train)+"/500] ACC TEST >= 0.95 ["+str(count_test)+"/500]")

    print(f'It took {end_time - start_time: 0.2f} second(s) to complete.')

    print("\nEnd demo \n")

# end script

