import numpy as np
import dill

class weightsNeurons:
    def __init__(self,inw, how,hog,ihg,ihpwd,hopwd):
        self.in_hid_weights = inw
        self.hid_out_weights = how
        self.hid_out_grads = hog
        self.in_hid_grads = ihg
        self.ih_prev_weights_delta = ihpwd
        self.ho_prev_weights_delta = hopwd


def save_weights(object, filename):
    dill.dump(object, file = open(filename, "wb"))

def load_weights(filename):
    return dill.load(open(filename, "rb"))