import numpy as np
from matplotlib import pyplot as plt


def predict(x, weights):
    x = np.concatenate(([1], x))
    
    neurons_level = [x]
    for weight_level in weights:
        previous_level = neurons_level[-1]
        neurons_level += [np.concatenate(([1],[previous_level.dot(w) for w in weight_level]))]
    print(neurons_level)


def main():
    weights = np.array([[[1,1,-1],
    [0,-1,-1],
    [-1,0,1]],
    [[1,-1,1,-1]]])

    x = np.array([1,1])
    predict(x, weights)
    pass

if __name__ == '__main__':
    main()