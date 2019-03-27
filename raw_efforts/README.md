

## Lab Note

* [2019.3.15]  
new direction: non-iid adversarial defence and attack  

* [2019.3.19]  
run vgg, resnet model, train on mnist, test on font-digits result:  
64, 56, 60, 58, 61, 70, 58, 64, 60  
53, 52, 49, 62, 61, 53, 48, 61, 60  
for training epochs 1 to 10  

* [2019.3.27]  
finish pick out > 90% data points    
vgg results in \~700 data > 90%, \~250 > 98%    
res results in \~40 data > 90%, \~3 > 98%  

vgg results:    
epoch 0, >90% 576, >95% 347,>98% 186,>99% 108,    
epoch 1, >90% 709, >95% 457,>98% 235,>99% 139,    
epoch 2, >90% 745, >95% 541,>98% 325,>99% 201,    
epoch 3, >90% 715, >95% 479,>98% 248,>99% 140,    
epoch 4, >90% 716, >95% 477,>98% 252,>99% 143,    
epoch 5, >90% 750, >95% 501,>98% 291,>99% 177,    
epoch 6, >90% 755, >95% 495,>98% 282,>99% 174,    
epoch 7, >90% 750, >95% 504,>98% 295,>99% 173,    
epoch 8, >90% 783, >95% 515,>98% 325,>99% 208,    
epoch 9, >90% 738, >95% 488,>98% 256,>99% 131,    

choose epoch 8
very imbalenced....  
num 0 total 19 >95%  
num 1 total 6 >95%  
num 2 total 113 >95%  
num 3 total 0 >95%  
num 4 total 89 >95%  
num 5 total 331 >95%  
num 6 total 7 >95%  
num 7 total 5 >95%  
num 8 total 0 >95%  
num 9 total 9 >95%  



TODO: 
3. vgg train n epoch on mnist test on mnist_test and font-digit, record percent
4. do FGSM on mnist_test and font-digit, record percent

next TODO:
* Question: does vgg give stable result for higher training epochs?
* reproduce normal defence
* reproduce certified defence

future TODO:
* should create non-iid data with different means