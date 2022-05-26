import numpy as np
import random

def takeRowsFromFile(file_name):
    rows = np.genfromtxt(file_name, dtype=np.float32, delimiter=',')
    return rows

def randomDataSets(file_name, per_sizeof_test):
    LearningSets = takeRowsFromFile(file_name)
    size = int(150 * per_sizeof_test)
    TestingSets = np.zeros(shape=[int(size), 5], dtype=np.float32)
    LearningSets = takeRowsFromFile(file_name)

    ran = int(size/3)

    idx_first = 0
    idx_sec = 0
    idx_third = 0
    for i in range(0, int(size)):
        a = 0
        b = 50
        if idx_first < 10:
            randNumber = random.randint(int(a), int(b) - int(idx_first))
            TestingSets[i] = np.add(TestingSets[i], LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)
            idx_first += 1
        elif idx_sec < 10:
            a = 50 - ran
            b = 100 - ran
            randNumber = random.randint(int(a), int(b) - int(idx_sec))
            TestingSets[i] = np.add(TestingSets[i],LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)
            idx_sec += 1
        elif idx_third < 10:
            a = 100 - (2*ran)
            b = 150 - (2*ran)

            randNumber = random.randint(int(a), int(b) - int(idx_third))
            TestingSets[i] = np.add(TestingSets[i],LearningSets[randNumber])
            LearningSets = np.delete(LearningSets, randNumber, axis=0)
            idx_third += 1
    return LearningSets, TestingSets
