from matplotlib import pyplot as plt
import numpy as np

def sigma(x):
    return 1/(1+np.exp(-x))

def plot_w(w, c='b'):
    x = np.linspace(-5, 5, 2)
    y = (-w[1] * x - w[0])/w[2]
    plt.plot(x, y, c)

def ex2_b():
    w = np.array([1,2,2])
    x = np.array([-1,1])
    x_wide = np.concatenate(([1], x))
    t = -1 # X is on the + side + is wrongly categorised -> t = -1
    eta = 0.6

    w_n = w - eta * (sigma(w.dot(x_wide)) - t) * x_wide

    plt.plot(*x, 'Xr')
    plot_w(w)
    plot_w(w_n, 'r')
    plt.show()
    
    print(*w_n)

def main():
    ex2_b()

if __name__ == '__main__':
    main()