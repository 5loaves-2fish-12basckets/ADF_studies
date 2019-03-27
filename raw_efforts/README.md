

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
vgg results in ~700 data > 90%, ~250 > 98%
res results in ~40 data > 90%, ~3 > 98%

TODO: 
1. pick out ~700 data, ~250 data for vgg, check 0~9 availability, compare shape with <90%, check accuracy
2. do FGSM on selected data, mnist
3. do distilled protection on selected data, mnist
4. 
