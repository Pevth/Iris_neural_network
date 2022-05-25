import numpy as np

class NeuronLayer:
    def __init__(self, neurons_number):
        self.amount = neurons_number
        self.neurons = np.zeros(shape=[self.amount], dtype=np.float32)
        self.biases = np.zeros(shape=[self.amount], dtype=np.float32)
        #----------
        self.biases_grads = np.zeros(shape=[self.amount], dtype=np.float32)
        self.error = np.zeros(shape=[self.amount], dtype=np.float32)
        self.prev_biases_delta = np.zeros(shape=[self.amount], dtype=np.float32)