import dill
class IOValues:
    def __init__(self, input, output):
        self.input = input
        self.output = output


def save_val(object, filename):
    dill.dump(object, file = open(filename, "wb"))

def load_val(filename):
    return dill.load(open(filename, "rb"))