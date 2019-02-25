
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
>This paper 
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
1. <img src="/doc/paper_review/tex/c87e803bc91881c3b1545e3dc49a776c.svg?invert_in_darkmode&sanitize=true" align=middle width=1583.3824907999997pt height=24.65753399999998pt/>Grad-CAM(x) > \kappa$ [Grad-Cam](https://arxiv.org/abs/1610.02391) doing gradients on a trained model? grid cells? unclear mechanism
3. How to defeat this?
4. Attention based attack vs defence is not evaluated!!



---
## Sparse DNNs with Improved Adversarial Robustness
[paper link](https://papers.nips.cc/paper/7308-sparse-dnns-with-improved-adversarial-robustness.pdf)
>This paper 

#### selected key points
1.
#### branching points
1.

## Neural Networks with Structural Resistance to Adversarial Attacks
[paper link](https://arxiv.org/abs/1809.09262)
>This paper 

#### selected key points
1.
#### branching points
1.

---

## Adversarial Diversity and Hard Positive Generation

[paper link](https://arxiv.org/abs/1605.01775)
>This paper 

#### selected key points
1.
#### branching points
1.

## Wild patterns: Ten years after the rise of adversarial machine learning

[paper link](https://www.sciencedirect.com/science/article/pii/S0031320318302565)
>This paper 

#### selected key points
1.
#### branching points
1.
