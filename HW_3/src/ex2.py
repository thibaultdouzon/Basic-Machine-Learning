from matplotlib import pyplot as plt
import numpy as np

def plot_identity():
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, 'b')
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('Random classifier ROC curve')
    plt.draw()
    plt.show()

def main():
    plot_identity()


    pass



if __name__ == '__main__':
    main()