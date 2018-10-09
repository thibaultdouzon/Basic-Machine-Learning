import numpy as np
from matplotlib import pyplot as plt


def sigma(x):
    return 1/(1+np.exp(-x))


def compute_out(w, d):
    hidden_layer = np.array([sigma(d.dot(input_weight)) for input_weight in w[0]])
    print(hidden_layer)
    hidden_layer_bias = np.concatenate(([1], hidden_layer))
    output_layer = np.array([hidden_layer_bias.dot(hidden_weight) for hidden_weight in w[1]])
    print(output_layer)


def main():
    weights = np.array([[[1,1,-1],[0,-1,-1], [-1,0,1]], 
                        [[1,-1,1,-1], [-1,1,-1,1]]])
    data = np.array([1,1])
    target = np.array([1,-1])

    compute_out(weights, np.concatenate(([1],data)))
    

if __name__ == '__main__':
    main()
