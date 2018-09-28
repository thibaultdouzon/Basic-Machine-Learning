import numpy as np
from matplotlib import pyplot as plt
from math import log

def entropy(x):
    assert(0<= x <= 1)
    if x==0 or x==1: return 0
    return -x*log(x,2)-(1-x)*log(1-x,2)

def best_split(data, label):
    tuples = list(zip(data, label))
    tuples = print(sorted(tuples))

    def compute_entropy(data, label, split):
        below_pos = [1 if x<split and l==1 else 0 for x,l in zip(data, label)].count(1)
        below_neg = [1 if x<split and l==0 else 0 for x,l in zip(data, label)].count(1)
        over_pos = [1 if x>split and l==1 else 0 for x,l in zip(data, label)].count(1)
        over_neg = [1 if x>split and l==0 else 0 for x,l in zip(data, label)].count(1)

        pos = below_pos + over_pos
        neg = below_neg + over_neg

        below = below_pos + below_neg
        over = over_pos + over_neg

        total = len(data)

        assert(total == below+over)
        assert(total == pos+neg)
        if below == 0:
            e_b = 0
        else:
            e_b = entropy(below_pos/below)*below/total 
        if over == 0:
            e_o = 0
        else:
            e_o = entropy(over_pos/over)*over/total
        
        print(f'SPLIT {split}, POS:({below_pos}, {over_pos}), NEG:({below_neg}, {over_neg}), RES:({e_b}, {e_o})')
        return e_b + e_o

    dataset_entropy = entropy([1 if l==1 else 0 for x,l in zip(data, label)].count(1)/len(data))

    splits = set()
    for value in sorted(data):
        splits.add((value-0.5, compute_entropy(data, label, value-0.5)))
        splits.add((value+0.5, compute_entropy(data, label, value+0.5)))

    print(dataset_entropy)
    for s, e in sorted(splits):
        print(f'Entropy for {s} split is: {e}, information gain: {dataset_entropy-e}')
def main():
    a3 =    [1,6,5,4,7,3,8,7,5]
    label = [1,1,0,1,0,0,0,1,0]
    best_split(a3, label)
    pass


if __name__ == '__main__':
    main()