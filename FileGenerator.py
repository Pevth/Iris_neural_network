import numpy as np
import random

def takeRowsFromFile(file_name):
    rows = np.genfromtxt(file_name, dtype=np.float32, delimiter=',')
    return rows

def randomDataSets(file_name):
    TestingSets = np.zeros(shape=[30, 5], dtype=np.float32)
    LearningSets = takeRowsFromFile(file_name)
    for i in range(0, 30):
        a = 0
        b = 50
        if i < 10:
            randNumber = random.randint(a, b-i)
            TestingSets[i] = np.add(TestingSets[i], LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)
        elif i >= 10 and i < 20:
            a = 40
            b = 90
            randNumber = random.randint(a, b - i)
            TestingSets[i] = np.add(TestingSets[i],LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)
        elif i >= 20 and i < 30:
            a = 80
            b = 130
            randNumber = random.randint(a, b - i)
            TestingSets[i] = np.add(TestingSets[i],LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)

    return LearningSets, TestingSets
