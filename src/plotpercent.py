

from matplotlib import pyplot as plt 
import numpy as np 

import glob

vgg_results = []
res_results = []

for file in glob.glob('output/vgg/*.npy'):
    vgg_results.append(np.load(file))

for file in glob.glob('output/res/*.npy'):
    res_results.append(np.load(file))


vgg_results = np.array(vgg_results)
vgg_final = vgg_results.mean(0)
res_results = np.array(res_results)
res_final = res_results.mean(0)
all_results = np.concatenate((vgg_results, res_results))
all_final = all_results.mean(0)

# print(np.shape(vgg_final))
# print(vgg_final)

def plot_hist(result, path):
    plt.hist(result, 20, facecolor='blue')
    plt.savefig(path)
    plt.close()
def plot_count(result, path):
    count = []
    x_label = []
    # sort = np.sort(result)
    # pick out 90 to 99%
    for i in range(10):
        tick = 0.9+0.01*i
        # print(result)
        count.append((result > tick).sum())
        x_label.append(tick)
    plt.plot(x_label, count)
    plt.savefig(path)
    plt.close()


for result, fn in zip([vgg_final, res_final, all_final], ['vgg','res', 'all']):
# for result, fn in zip([res_final], ['res']):
    plot_hist(result, 'output/plt/%shist.png'%fn)
    plot_count(result, 'output/plt/%scount.png'%fn)

print('==all done==')