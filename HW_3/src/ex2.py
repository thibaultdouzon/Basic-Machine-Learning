from matplotlib import image as img
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

def plot_equal_error():
    image = img.imread('HW_3/img/ex2d.png')
    plt.imshow(image)
    
    points = [[50.3264, 30.5042], [356.812, 325.483]]
    plt.plot(*zip(*points))
    inter = [[121.456], [99.5418]]
    plt.scatter(*inter, marker='s')
    plt.legend(['Antidiagonal', 'Intersection point'])
    pass
    plt.show()

    roc_point = [(i[0]-b)/(e-b) for i, (b,e) in zip(inter, list(zip(*points)))]
    roc_point[1] = 1 - roc_point[1]
    print(*roc_point)


def main():
    # plot_identity()
    plot_equal_error()

    pass



if __name__ == '__main__':
    main()