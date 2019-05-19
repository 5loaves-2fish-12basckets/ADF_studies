### code construction plan

structure:
1. datafunc.py handles data
2. model.py handles model
3. trainer.py imports datafunc.py, model.py to train DCGAN
4. confic.py sets configurations in datafunc, model, and trainer

* stage 1
finish executable module for DCGAN on mnist
* stage 2
follow original paper to perform various experiments on model structure:
	1. pooling vs no pooling
	2. batchnorm vs no batchnorm
	3. ReLU(G), LeakyRelu(D) vs sigmoid
run on LSUN or imagenet-1k
* stage 3
follow medium article to perform on cifar10
* stage 4
follow original paper to classify on cifar10

```
experiment considerations for config:
mnist best model x M epochs => good result
check model' x M epochs => what result?
check best model x M epochs (k:1) => what result?
							(1:k)
```


## Error log
model parameters in this is too small and does not work for MNIST at all!
https://gist.github.com/xmfbit/cbdef5d6bfcb4f35f9c851161191f4b4
changed to 
https://github.com/togheppi/DCGAN/blob/master/MNIST_DCGAN_pytorch.py