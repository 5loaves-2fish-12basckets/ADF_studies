# Tensorflow tutorial

following https://www.tensorflow.org/tutorials

## Log

* abort plan to trace everything, aim to learn tensorflow

## Issues

* [2019 Feb. 20]
It seems that tensorflow fails when GPU only has 10% memory available
```
InternalError: Blas GEMM launch failed : a.shape=(10, 9), b.shape=(9, 64), m=10, n=64, k=9
	 [[{{node dense/MatMul}} = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](_arg_dense_input_0_0/_21, dense/MatMul/ReadVariableOp)]]
Or:
GPU Sync error
```
when trying 1_learn_and_use_ML/Regression up to 
{
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
}

==> solotion: tensorflow memory allocation
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.4
```
from [tensorflow using gpu guide](https://www.tensorflow.org/guide/using_gpu)

(side note) add the following for directly using (old version?) keras:
```
from keras import backend as k
k.tensorflow_backend.set_session(tf.Session(config=config))
```

* [2019 Feb. 22]
For some reason Regression is not working as supposed to
```
learning curve is not same, final prediction for testing data results in flat line
```
==> compare with tutorial to find difference....