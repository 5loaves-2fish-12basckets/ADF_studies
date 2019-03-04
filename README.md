# ADF_studies
this is a 21 week plan to study Adversarial Attacks and Defences

>paper reviews, tf-tutorials, and final report on GAN in /doc directory  
>reproduced models (in python) are in /src directory



### weekly schedule:
<details open><summary>finished progress</summary>

* week 1  
scheduled: Feb. 15 - 21 (semester starts)  
    
    1. read basics:
    Are Adversarial Examples inevitable? https://openreview.net/pdf?id=r1lWUoA9FQ  
    Adversarial Examples: Attacks and Defenses for Deep Learning https://arxiv.org/pdf/1712.07107.pdf  
    Explaining and Harnessing Adversarial Examples https://arxiv.org/abs/1412.6572  
    Intriguing properties of neural networks  https://arxiv.org/abs/1312.6199  
    Towards Evaluating the Robustness of Neural Networks https://arxiv.org/abs/1608.04644  
    Towards Deep Learning Models Resistant to Adversarial Attacks https://arxiv.org/abs/1706.06083  

<p align="right">  
result: <a href="doc/paper_review/week01.md">paper review</a><br>
date: Feb. 19
</p>


* week 2  
scheduled: Feb. 22 - 28  

    1. follow references and citations of previous papers   
    2. write a review on these papers  

<p align="right">  
result: <a href="doc/paper_review/week02.md">paper review</a><br>
date: Feb. 26
</p>
</details>

---

* week 3  
scheduled: Mar. 1 - 7  

1. read  (investigate certified robustness and provable defense)
Certified Robustness to Adversarial Examples with Differential Privacy  
https://arxiv.org/pdf/1802.03471.pdf  
Towards Fast Computation of Certified Robustness for ReLU Networks  
https://arxiv.org/pdf/1804.09699.pdf  
Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope  
https://arxiv.org/pdf/1711.00851.pdf  
Certified Robustness to Adversarial Examples with Differential Privacy  
https://arxiv.org/pdf/1802.03471.pdf  

also 
Wild patterns: Ten years after the rise of adversarial machine learning  
https://www.sciencedirect.com/science/article/pii/S0031320318302565  

2. reproduce MWD in my own best practice.
https://github.com/rakshit-agrawal/mwd_nets
https://github.com/lucadealfaro/rbfi

3. understand code  
code for certified robustness for relu networks  
https://github.com/huanzhang12/CertifiedReLURobustness  


* week 4  
scheduled: Mar. 8 - 14  

    1. follow references and citations of previous papers   
    2. write a review on these papers  
    3. follow tensorflow tutorials to learn tensorflow for future needs:  
https://www.tensorflow.org/tutorials/  
—> do generative models  

* week 5  
scheduled: Mar. 15 - 21  
(investigate NIPS challange)  
    1. read   
Adversarial Attacks and Defences Competition  
https://arxiv.org/pdf/1804.00097.pdf  
Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser  
https://arxiv.org/abs/1712.02976  
Mitigating Adversarial Effects Through Randomization  
https://arxiv.org/pdf/1711.01991.pdf  
(code https://github.com/cihangxie/NIPS2017_adv_challenge_defense)  
Boosting Adversarial Attacks with Momentum  
https://arxiv.org/pdf/1710.06081.pdf  
(code https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks)  
Ensemble Adversarial Training: Attacks and Defenses  
https://arxiv.org/abs/1705.07204  
(code https://github.com/sangxia/nips-2017-adversarial)  


* week 6  
scheduled: Mar. 22 - 28  
    1. follow references and citations of previous papers   
    2. write a review on these papers  

* week 7  
scheduled: Mar. 29 - Apr. 4 (there is holiday)  
    1. read  
universal adversarial patch   
https://medium.com/deep-dimension/deep-learning-papers-review-universal-adversarial-patch-a5ad222a62d2  
https://arxiv.org/pdf/1712.09665.pdf  
decision based black box attack  
https://openreview.net/pdf?id=SyZI0GWCZ  
spacially transform adversarial examples https://openreview.net/pdf?id=HyydRMZC-  

    2. finalize previous reviews and finish writing review paper  
  

* week 8  
scheduled: Apr. 5 - 11  (prepare midterm)

    1. reproduce Boosting Adversarial Attacks with Momentum (in pytorch)  
  
* week 9  
scheduled: Apr. 12 - 18 (Midterm)  
    1. reproduce certified robustness (in pytorch)  
test Boosting Adversarial Attacks with Momentum against certified robustness  

\<full review of adversarial attacks and simple reproductions>  
* week 10  
scheduled: Apr. 19 - 25  

* week 11  
scheduled: Apr. 26 - May 2  
「  
Goal:   
(following certified robustness)  
try to come up with best defence against state of the art adversarial attack on mnist dataset  

* week 12  
scheduled: May 3 - 9   


* week 13  
scheduled: May 10 - 16  

  
* week 14  
scheduled: May 17 - 23  

* week 15  
scheduled: May 24 - 30  
* week 16  
scheduled: May 31 - Jun. 6  
* week 17  
scheduled: Jun. 7 - 13  
* week 18  
scheduled: Jun. 14 - 20 (final) 

」  

\<come up with new idea and execute>  
* week 19  
scheduled: Jun. 21 - 27  
finish ADF results, write report  
* week 20  
scheduled: Jun. 28 - Jul. 4   
* week 21  
scheduled: Jun. 5 - 11  

finish ADF results, write report  

if there is time  
try  
https://medium.com/syncedreview/ai-brush-new-gan-tool-paints-worlds-2544e4e29c11  

\<wrap up for conclution>  


(因為找不到Adversarial attack 發展過程完善的整理，因此從最新的論文開始實作，依序往回找參考文獻，在自己整理出發展過程）  
參考列表：    
https://www.robust-ml.org/defenses/    
https://github.com/IBM/adversarial-robustness-toolbox    

https://github.com/yenchenlin/awesome-adversarial-machine-learning    
https://medium.com/@wielandbr/reading-list-for-the-nips-2018-adversarial-vision-challenge-63cbac345b2f    
https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html    

conference list    
https://aaai18adversarial.github.io/    
