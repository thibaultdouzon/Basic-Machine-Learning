import itertools as it
import numpy as np
from matplotlib import pyplot as plt
import scipy as sc
from scipy import io
import math
import random


def create_gauss(d: np.ndarray):
    return Gauss(d.mean(axis=0), np.cov(d.T))


class Gauss:
    def __init__(self, mu, sigma):
        """Initialise a distribution with mean mu and covariance sigma

        Precompute and store everything that is not dependent
        on the datapont, so as to keep things efficient"""
        D = mu.size
        self.mu = mu
        if D > 1:
            self.icov = np.linalg.inv(sigma)
            sign, ld = np.linalg.slogdet(sigma)
            if sign != 1:
                print("Sign=", sign)

            self.lognum = D*np.log(2*np.pi) + ld
        else:
            self.icov = 1/sigma
            self.lognum = D*np.log(2*np.pi*sigma)

    def logprob(self, x):
        """return log(p(x))"""
        d = x-self.mu
        return -.5 * (self.lognum + np.dot(np.dot(d, self.icov), d))

    def prob(self, x):
        """return p(x)"""
        return np.exp(self.logprob(x))


def predict(x, gauss_l, prior_l):
    res = []
    evidence = sum([g.prob(x)*p for g, p in zip(gauss_l, prior_l)])
    for c in [0, 1]:
        likeli = gauss_l[c].prob(x)
        res += [likeli*prior_l[c]/evidence]
    return res


def conf_mat(gauss_l, prior_l, d, f):
    """
    TP | FP
    ---+---
    FN | TN
    """
    mat = np.array([[0, 0], [0, 0]])

    pred = [predict(x, gauss_l, prior_l).index(
        max(predict(x, gauss_l, prior_l))) for x in f(d)]
    real = d[:, -1]/2-1

    for i in range(2):
        for j in range(2):
            mat[i, j] = len(
                list(filter(lambda x: x[0] == i and x[1] == j, zip(pred, real))))

    precision = mat[1, 1]/(mat[1, 1]+mat[1, 0])
    recall = mat[1, 1]/(mat[1, 1]+mat[0, 1])
    print(mat)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')


def evaluate(gauss_l, prior_l, f):
    """
    report error on training AND validation set
    confusion matrix, precision andd recall for malignant class
    """
    d_t = io.loadmat(
        "HW_6/breast_cancer/breast_cancer_train.mat")['breast_cancer_train']
    d_v = io.loadmat(
        "HW_6/breast_cancer/breast_cancer_validation.mat")['breast_cancer_validation']

    print("Confusion matrix TRAIN")
    conf_mat(gauss_l, prior_l, d_t, f)
    print("Confusion matrix VALIDATION")
    conf_mat(gauss_l, prior_l, d_v, f)

    pass


def plot_q1(d):
    plt.subplot('221')
    plt.hist(d[:, 3])
    plt.subplot('223')
    plt.hist(np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 3])
    plt.subplot('224')
    plt.hist(np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 3])
    plt.show()
    pass


def q1(d):
    """
    index 3 only
    plot paramaters
    """
    print("-"*10)
    print("Q1")
    # plot_q1(d)
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 3]
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 3]

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x: x[:, 3])

    pass


def q2(d):
    """
    First ACP
    plot paramaters
    """
    print("-"*10)
    print("Q2")
    u, _, _ = np.linalg.svd(np.cov(d[:, 1:-1].T))
    w = u[:, 0]
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 1:-1]@w
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 1:-1]@w

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x, w=w: x[:, 1:-1]@w)


def q3(d):
    """
    index 3 and 9
    plot paramaters
    """
    print("-"*10)
    print("Q3")
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, (3, 9)]
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, (3, 9)]

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x: x[:, (3, 9)])


def q4(d):
    """
    First two ACP
    plot paramaters
    """
    print("-"*10)
    print("Q4")
    u, _, _ = np.linalg.svd(np.cov(d[:, 1:-1].T))
    w = u[:, 0:2]
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 1:-1]@w
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 1:-1]@w

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x, w=w: x[:, 1:-1]@w)


def q5(d):
    """
    80% acp
    plot paramaters
    """
    print("-"*10)
    print("Q5")
    u, v, _ = np.linalg.svd(np.cov(d[:, 1:-1].T))
    tot_var = sum(v)
    cumul = 0
    for i, x in enumerate(v):
        cumul += x
        if cumul > 0.8*tot_var:
            break
    w = u[:, 0:i+1]
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 1:-1]@w
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 1:-1]@w

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x, w=w: x[:, 1:-1]@w)


def q6(d):
    """
    index all
    plot paramaters
    """
    print("-"*10)
    print("Q6")
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, 1:-1]
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, 1:-1]

    gauss_l = [create_gauss(x) for x in [A, B]]
    prior_l = [len(list(filter(lambda x:x[-1] == i, d)))/len(d)
               for i in [2, 4]]
    print("A")
    print(A.mean(axis=0), np.cov(A.T))
    print("B")
    print(B.mean(axis=0), np.cov(B.T))
    evaluate(gauss_l, prior_l, lambda x: x[:, 1:-1])


def ex5(d):
    def sigma(x):
        return 1/(1+np.exp(-x))

    def predict_log(w, x):
        return sigma(w.dot(x))

    def predict_log_class(w, x):
        return round(predict_log(w, x))

    def error_log(w, x, t):
        return t*np.log2(predict_log(w, x)) + (1-t)*np.log2(1-predict_log(w, x))

    def grad_log(w, x, t):
        return (predict_log(w, x)-t)*x

    def conf_mat_log(w, d):
        """
        TP | FP
        ---+---
        FN | TN
        """
        mat = np.array([[0, 0], [0, 0]])

        pred = [predict_log_class(w, np.concatenate(([1], x)))
                for x in d[:, 1:-1]]
        real = d[:, -1]/2-1

        for i in range(2):
            for j in range(2):
                mat[i, j] = len(
                    list(filter(lambda x: x[0] == i and x[1] == j,
                                zip(pred, real))))

        precision = mat[1, 1]/(mat[1, 1]+mat[1, 0])
        recall = mat[1, 1]/(mat[1, 1]+mat[0, 1])
        print(mat)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

    def plot_roc(w, d):
        roc = []
        for t in np.linspace(0, 1, 100):
            mat = np.array([[0, 0], [0, 0]])

            pred = [0 if predict_log(w, np.concatenate(([1], x))) < t else 1
                    for x in d[:, 1:-1]]
            real = d[:, -1]/2-1

            for i in range(2):
                for j in range(2):
                    mat[i, j] = len(
                        list(filter(lambda x: x[0] == i and x[1] == j,
                                    zip(pred, real))))
            fpr = 0
            if mat[0][0]+mat[1][0] > 0:
                fpr = mat[1][0]/(mat[0][0]+mat[1][0])
            tpr = 0
            if mat[0][1]+mat[1][1] > 0:
                tpr = mat[1][1]/(mat[0][1]+mat[1][1])
            roc += [[tpr, fpr]]

        roc = np.array(roc)
        plt.axis(aspect='equal')
        plt.plot(roc[:, 1], roc[:, 0])
        plt.xlabel('FPrate')
        plt.ylabel('TPrate')
        plt.show()
        roc = np.array(list(reversed(roc)))
        roc = np.array([roc[:,1], roc[:,0]]).T
        area = sum([(roc[i+1, 0]-roc[i, 0])*(roc[i+1, 1] + roc[i, 1])/2 for i in range(len(roc)-1)])
        print(area)

    def evaluate_log(w):
        """
        report error on training AND validation set
        confusion matrix, precision andd recall for malignant class
        """
        d_t = io.loadmat(
            "HW_6/breast_cancer/breast_cancer_train.mat")['breast_cancer_train']
        d_v = io.loadmat(
            "HW_6/breast_cancer/breast_cancer_validation.mat")['breast_cancer_validation']

        print("Confusion matrix TRAIN")
        conf_mat_log(w, d_t)
        print("Confusion matrix VALIDATION")
        conf_mat_log(w, d_v)

        plot_roc(w, d_v)

    def stochastic(w, d, eta, l, N):
        n = 0
        while n < N:
            di = random.choice(d)
            w = w - eta * grad_log(w, np.concatenate(([1], di[1:-1])), di[-1]/2-1) - np.concatenate(([0], l*w[1:]))
            n += 1
        return w

    w = np.ones(10)
    w = stochastic(w, d, 0.005, 0.001,  10000)
    print(f'Final discriminant: {w}')
    print(len(d))
    evaluate_log(w)
    pass


def main():
    d = io.loadmat("HW_6/breast_cancer/breast_cancer_train.mat")
    d = d['breast_cancer_train']
    # q1(d)
    # q2(d)
    # q3(d)
    # q4(d)
    # q5(d)
    # q6(d)
    # ex5(d)
    # for feat in range(1, 11):
    #     plt.subplot('221')
    #     plt.hist(d[:,feat])
    #     plt.subplot('223')
    #     plt.hist(np.array(list(filter(lambda x:x[-1]==2,d)))[:,feat])
    #     plt.subplot('224')
    #     plt.hist(np.array(list(filter(lambda x:x[-1]==4,d)))[:,feat])
    #     plt.show()
    from scipy.stats import norm
    
    A = np.array(list(filter(lambda x: x[-1] == 2, d)))[:, (9)]
    B = np.array(list(filter(lambda x: x[-1] == 4, d)))[:, (9)]

    print(B)
    muA = A.mean(axis=0)
    sigA = math.sqrt(np.cov(A.T))
    xA = np.linspace(1,10,1000)
    yA = norm.pdf(xA, muA, sigA)

    muB = B.mean(axis=0)
    sigB = math.sqrt(np.cov(B.T))
    xB = np.linspace(1,10,1000)
    yB = norm.pdf(xB, muB, sigB) 
    plt.figure('Feature 10: Micoses')
    plt.subplot('121')
    plt.hist(np.array(list(filter(lambda x:x[-1]==2,d)))[:,9], normed=True)
    plt.plot(xA, yA, '--r')
    plt.legend(['Gaussian approximation'])
    plt.title('Benign')
    plt.subplot('122')
    plt.hist(np.array(list(filter(lambda x:x[-1]==4,d)))[:,9], normed=True)
    plt.plot(xB, yB, '--r')
    plt.legend(['Gaussian approximation'])
    plt.title('Malignant')
    plt.show()

    pass


if __name__ == '__main__':
    main()
