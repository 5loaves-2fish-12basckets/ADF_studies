
import json
from statistics import mean
import numpy as np

from matplotlib import pyplot as plt

with open('small_data.json', 'r') as f:
    DATA = json.load(f)

'''
DATA
    from
    to 
    samples
        label1  int
        label2  int
        distances   list of 4 float

mtrain  mtrain  mtrain  mtest   mtest   font
mtrain  mtest   font    mtest   font    font
'''

## 1.
## want to know mtrain <-> mtrain, mtest, vs font :: min, max, avg
## mtest <-> font
def write_log(sentence):
    print(sentence)
    with open('log.txt', 'a') as f:
        f.write(sentence)


def plot_hist(result,title, path):
    plt.hist(result, 20, facecolor='blue')
    plt.title(title)
    plt.savefig(path)
    plt.close()

metrix_list = ['l0', 'l1', 'l2', 'l8']
choose = [1,2,3, 5]
set_list0 = ['mnist-train', 'mnist-train', 'mnist-train', 'mnist-test']
set_list = ['mnist-train', 'mnist-test', 'font-digit', 'font-digit']

write_log('COARSE INFO')
write_log(' - - - - - ')

for i in range(4):
    # collect distances
    print()
    message = 'result for %s distance'%metrix_list[i]
    write_log(message)
    for j in range(4):
        choose_set = choose[j]
        collect = []
        for sample in DATA[choose_set]['samples']:
            if sample['distances'][i] != 0:
                collect.append(sample['distances'][i])

        title = '%s_%s'%(set_list0[j], set_list[j])
        write_log(title)
        write_log('min %f max %f avg %f'%(min(collect), max(collect), mean(collect)))
        plot_hist(collect, title, title+'.png')
        write_log('===')

write_log('FINE INFO')
write_log(' - - - - - ')

## 1.1
## want to know mtrain [0] <-> mtrain, mtest, vs font :: min, max, avg
## mtest <-> font

lengths = [i for i in range(9, 0, -1)]

for i in range(4):
    # collect distances
    print('result for %s distance'%metrix_list[i])
    for j in range(4):
        choose_set = choose[j]
        collect = [[] for __ in range(55)]
        for sample in DATA[choose_set]['samples']:
            a = sample['label1']
            b = sample['label2']
            if a > b :
                a , b = b, a
            k = sum(lengths[:a]) + b

            if sample['distances'][i] != 0:
                collect[k].append(sample['distances'][i])
        collect = np.array(collect)

        title = '%s to %s'%(set_list0[j], set_list[j])
        print(title)


        plot_hist(collect, title, title+'.png')

        



## 2.
## want to plot mtrain <-> mtrain, mtest, vs font
## mtest <-> font
