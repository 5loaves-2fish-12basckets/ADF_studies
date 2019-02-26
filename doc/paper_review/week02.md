
# week2 paper reviews for ADF
scheduled date: Feb. 8 - Feb. 14    

1. contour context
2. nonlinearity
3. other topic continuing prev week papers
> paper list1:  
[Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers](#Extending-Adversarial-Attacks-and-Defenses-to-Deep-3D-Point-Cloud-Classifiers)
[Attention, Please! Adversarial Defense via Attention Rectification and Preservation](#Attention-Please-Adversarial-Defense-via-Attention-Rectification-and-Preservation)

> paper list2:  
[Sparse DNNs with Improved Adversarial Robustness](#Sparse-DNNs-with-Improved-Adversarial-Robustness)
[Neural Networks with Structural Resistance to Adversarial Attacks](#Neural-Networks-with-Structural-Resistance-to-Adversarial-Attacks)

> paper list3:
> (quick check on PASS)
> [Adversarial Diversity and Hard Positive Generation](#Adversarial-Diversity-and-Hard-Positive-Generation)
>(check if Towards deep learning ... is deafeated as of 2018)
>[Wild patterns: Ten years after the rise of adversarial machine learning](#Wild-patterns-Ten-years-after-the-rise-of-adversarial-machine-learning)
## Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers 
[paper link](https://arxiv.org/abs/1901.03006)
>This paper investigates adversarail attack and defence on 3D Point Cloud Classifiers, and found it to be attackable, and also defendable to some degree
>(Jan. 2019)

#### selected key points
1. Point cloud classifiers classifies 'point cloud data' i.e. a bunch of points forming 3D representation. The only info is position, therefore adversarial samples moves points around and is more easily detectible due to outliers that change **overall shape**. Can be prevented by a. projecting to surface, and b. clip norm of neighboring distance.
2. Defend by a. adversarial training (Ian Goodfellow) b. Remove outlier (high avg dist. to neighbor); Remove salient point 
3. Attacks are successful (~90%), but defence can break attacks success rate in half
#### branching points
1. If deep learning is as good as human brain, than adversarial examples that does not change **overall shape** should have no effect, and those with few outliers should have minimum effect.
2. In (7. Discussion): max-pooling operations ... hide a subset of pts so thay have zero gradients and could not be perturbed ... and after outliers, salient points removed (defence procedures) those points are exposed and can represent overall shape of point cloud
3. Best defensive method is by removing outliers (hinting on **overall shape?**; data is not provided)

## Attention, Please! Adversarial Defense via Attention Rectification and Preservation
[paper link](https://arxiv.org/abs/1811.09831)
>This paper applies insights gained from attention analysis on deep learning models to build strong adversarial defence (that does not deviate attention much when subject to adversarial example) and also apply to new attack (that considers attention)
>(Nov. 2018)

#### selected key points
1. Prev. defences mostly within three group: (a.) Denoising, transforming adversarial sample before feeding into the model (b.) Model modification and (c.) Adversarial training
2. (c.) is applicable to various different algorithms. It has 3 factors: (i) focus on robust features [Tsipras et al.](https://arxiv.org/abs/1805.12152) found that adversarial attack amplify importance of low confidence features (ii) reduce feature distribution divergence (iii) more robust if train on stronger advdersarial examples
3. visual attention -> region of interest. Successful attack deviates and scatters the attention area
4. Attention-baseed Adversarial Defense (AAD): (i)rectification, correct the classifcation? (ii) perservation, align attention area btw adversarial and original imgs (iii) selection of strong adversarial samples based on attention
5. IoU(truth, attention)<0.5 would likely be attacked.  Samples on which attack fails have higher IoU
6. IoU(ori, adv) small for attacks that succeed.
7. iter attack can shrink and scatter IoU step by step, but after (2) iteration IoU continue to drop but success rate will not climb, and adversarial example will become noticeable (which is bad for attack)

##### 3 part AAD model
1. rectification
help model focus attention on groud truth area (unavailable) --> rectify completeness of attention area (not scattered) by: <p align="center"><img src="/doc/paper_review/tex/67c7802ba5bfcb5a76de9c251df1efb5.svg?invert_in_darkmode&sanitize=true" align=middle width=160.76281154999998pt height=16.438356pt/></p> <p align="center"><img src="/doc/paper_review/tex/fcbf666e1fe006969c45ed112da08759.svg?invert_in_darkmode&sanitize=true" align=middle width=232.53807224999997pt height=16.438356pt/></p>
2. perservation
reduce shifting of attention for adversarial example. <p align="center"><img src="/doc/paper_review/tex/012fae5bdc99fd0f6669c91e8f6c5792.svg?invert_in_darkmode&sanitize=true" align=middle width=226.27539495pt height=16.438356pt/></p>This will give total loss <p align="center"><img src="/doc/paper_review/tex/6d5baf5f91e7321f549ae91edd1675a5.svg?invert_in_darkmode&sanitize=true" align=middle width=248.50941224999997pt height=17.031940199999998pt/></p>

3. select strong adversarial sample (that deviates attention more)

##### result
1. against StepLL, R+StepLL, Iter-LL have improvements better than [Madry et al.](https://arxiv.org/abs/1706.06083) (last paper of week1! Towards..., was state of the art)

#### more key points
1. <img src="/doc/paper_review/tex/9883d68e9a70fbbaf4f2f1e833e55165.svg?invert_in_darkmode&sanitize=true" align=middle width=44.777676899999996pt height=22.831056599999986pt/> ratio is different in MNIST and cifar10 setting (8,4,1 is generally best)
2. applied to attack by adding term to encourage attention deviation
#### branching points
1. A production of Beijing, China
2. Attention area <img src="/doc/paper_review/tex/1cc8338ace24e38a43eebc16e87aff6e.svg?invert_in_darkmode&sanitize=true" align=middle width=222.99527084999994pt height=24.65753399999998pt/> 
Check out [Grad-Cam](https://arxiv.org/abs/1610.02391)! 
3. How to defeat this?
4. Attention based attack vs defence is not evaluated!!



---
## Sparse DNNs with Improved Adversarial Robustness
[paper link](https://papers.nips.cc/paper/7308-sparse-dnns-with-improved-adversarial-robustness.pdf)
>Pruning a model saves computation/memory costs. This paper suggests that it can also help defend adversarial attack and discusses inefficiency and unrobustness of DNN together.

#### selected key points
1. DNN (basically deep learning) has redundent feature representations, and could improve for mobile application "[Predicting parameters in deep learning](https://arxiv.org/abs/1306.0543)" On the other hand there is the adversarial issue.
2. sparsity of connection or sparsity of neuron activity

!! up to 3 !! many math...

#### branching points
1. Is a sparse DNN equivalent to a non linear DNN as coined in this paper?
2. The conclution here seems to be coherent with the paper above that adversarial attack utilizes less confidence connections/predictions/areas

## Neural Networks with Structural Resistance to Adversarial Attacks
[paper link](https://arxiv.org/abs/1809.09262)
>This paper promotes RBFI to replace ReLu! Following GoodFellow et al. (2014) that states local linearity may be key, the author aims to create a highly non-linear deep leaning model.

#### selected key points
1. (linearity issue) <img src="/doc/paper_review/tex/a5f0bd69955128fc5fe34f42a7365fb5.svg?invert_in_darkmode&sanitize=true" align=middle width=261.19126769999997pt height=26.438629799999987pt/> will have output perturbation of <img src="/doc/paper_review/tex/f593d569df8fde0356aac100f399c5f4.svg?invert_in_darkmode&sanitize=true" align=middle width=77.99382194999998pt height=26.438629799999987pt/> which will be huge for large n
2. (gaussian) radial basis function is <img src="/doc/paper_review/tex/b205da1e6227c577cc3e7931d563d93c.svg?invert_in_darkmode&sanitize=true" align=middle width=100.35654749999999pt height=32.44583099999998pt/> RBFI follows this and adds (i) remove radial symmetry and allow RBFIs to scale each component (sensitivity) individually; (ii) calculate distance not in <img src="/doc/paper_review/tex/7252ad06a4944da2b6628a58281cb887.svg?invert_in_darkmode&sanitize=true" align=middle width=11.45742179999999pt height=22.831056599999986pt/> but in <img src="/doc/paper_review/tex/2b4f8dfb585beeba8b6047b353d0efd9.svg?invert_in_darkmode&sanitize=true" align=middle width=18.00995789999999pt height=22.831056599999986pt/>, which is maximum of difference, so we have n=1. (I in RBFI is infinity)
3. pseudo gradient!
4. result: similar to relu/sigmoid for MNIST, superb for adversarial MNIST (90% vs 2%)!

!! up to 2 !! finish this + reproduce!
#### branching points
1. reproduce this!!!https://github.com/lucadealfaro/rbfi

---

## Adversarial Diversity and Hard Positive Generation

[paper link](https://arxiv.org/abs/1605.01775)
>We are focusing on PASS in this paper 

#### selected key points
1.
#### branching points
1.

## Wild patterns: Ten years after the rise of adversarial machine learning

[paper link](https://www.sciencedirect.com/science/article/pii/S0031320318302565)
>We are focusing on "Towards deep leaning..." in this paper

#### selected key points
1.
#### branching points
1.
