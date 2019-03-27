
import glob
import numpy as np 

allresults = [[] for __ in range(10)]
for file in glob.glob('output/vgg/*.npy'):
    ep = int(file.split('-')[1][0])
    allresults[ep].append(np.load(file))

for i in range(10):
    epresult = np.array(allresults[i])
    epresult = epresult.mean(0)
    ep90 = (epresult>0.9).sum()
    ep95 = (epresult>0.95).sum()
    ep98 = (epresult>0.98).sum()
    ep99 = (epresult>0.99).sum()
    print('epoch %d, >90%% %d, >95%% %d,>98%% %d,>99%% %d,'%(i, ep90, ep95, ep98, ep99))
