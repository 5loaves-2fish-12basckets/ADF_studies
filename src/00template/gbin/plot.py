from matplotlib import pyplot as plt 
import json

with open('ckpt/result.json', 'r') as f:
    result = json.load(f)

x = [i*0.01 for i in range(10)]
for name in ['lin', 'wide', 'deep']:
    b1, b2, c1, c2 = result[name]
    plt.title(name)
    plt.plot(x,b1, label='base_fgsm')
    plt.plot(x,b2, label='base_pgd')
    plt.plot(x,c1, label='cert_fgsm')
    plt.plot(x,c2, label='cert_pgd')
    plt.legend()
    plt.savefig(name+'.png')
    plt.close()