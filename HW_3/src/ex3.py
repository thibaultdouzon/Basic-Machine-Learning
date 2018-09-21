import numpy as np
from matplotlib import pyplot as plt

def sigma(x):
    return 1/(1+np.exp(-x))

def gradient(w, data, label):
    pass
    
def stochastic(w, data, label):
    pass

def main():
    w0 = np.array([0,1,-2,2])
    data = np.array([[1,-1,-2]])
    label = np.array([[1]])
    print(w0.dot(np.concatenate(([1], data[0]))))
    pass

if __name__ == '__main__':
    main()