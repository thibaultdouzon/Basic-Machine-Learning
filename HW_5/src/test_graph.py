
from itertools import product

labels = ['input layer', 'hidden layer 1', 'hidden layer 2', 'output layer']
layers = [('i', 4), ('h', 6), ('hh', 6), ('o', 3)]
cols = {'i': 'white', 'h': 'grey50', 'hh': 'gray50', 'o': 'gray80'}

print('graph nn {')
print('  rankdir=LR;')
print('  ranksep="0.8 equally";')
print('  nodesep="0.05 equally";')
print('  splines=line;')

for layer, label in zip(layers, labels):
    name = layer[0]
    col = cols[name]
    print('  subgraph cluster_%s {' % name)
    print('    style=filled;')
    print('    peripheries=0;')
    print('    label="%s";' % label)
    print('    node [label="",style=filled,shape=circle,fillcolor=%s];' % col)
    for n in range(layer[1]):
        print('    %s%d' % (name, n))
    print('  }')
for a, b in zip(layers, layers[1:]):
    for n1, n2 in product(range(a[1]), range(b[1])):
        print('  %s%d -- %s%d;' % (a[0], n1, b[0], n2))
print('}')