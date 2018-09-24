import numpy as np
from matplotlib import pyplot as plt

def sigma(x):
    return 1/(1+np.exp(-x))

def gradient(w, data, label):
    """
    :param w: Discriminant
    :param data: One point, extended such that len(data) == len(w)
    :param label: Label of the data, 0 or 1
    """
    return (sigma(w.dot(data)) - label) * data

    
def stochastic(w, data, label, eta, lambda_, L):
    data = np.concatenate(([1], data))
    print(L(w))
    print(f'gradient = {gradient(w, data, label)}')
    print(f'regularization = {L(w)}')
    w_new = w - eta * gradient(w, data, label) - lambda_ * L(w)
    return np.array(w_new)


def L1(w):
    return np.array(np.concatenate(([0], [1 if x > 0 else (-1 if x < 0 else 0) for x in w[1:]])))

def L2(w):
    return np.array(np.concatenate(([0], w[1:])))


def main():
    w0 = np.array([0,1,-2,2])
    data = np.array([[1,-1,-2]])
    label = np.array([1])
    print(w0.dot(np.concatenate(([1], data[0]))))

    w_l0 = stochastic(w0, data[0], label[0], 0.7, 0.2, lambda x: 0)
    print(f'No regularization : {w_l0}')
    print(w_l0.dot(np.concatenate(([1], data[0]))))

    w_l1 = stochastic(w0, data[0], label[0], 0.7, 0.2, L1)
    print(f'L1 regularization : {w_l1}')
    print(w_l1.dot(np.concatenate(([1], data[0]))))
    
    w_l2 = stochastic(w0, data[0], label[0], 0.7, 0.2, L2)
    print(f'L2 regularization : {w_l2}')
    print(w_l2.dot(np.concatenate(([1], data[0]))))    
    pass

def roc():
    x = np.linspace(0, 1, 100)
    a = x**2
    b = x**0.38
    c = x**0.5 + 0.2*np.sin(x*3*np.pi)/np.exp(1.5*x)
    # plt.plot(x,x)
    plt.plot(x, a)
    plt.plot(x, b)
    plt.plot(x, c)
    plt.plot([0,0,1], [0,1,1])

    plt.legend(['a','b','c','d'])
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('Comparing ROC curves')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    roc()
    # main()

    