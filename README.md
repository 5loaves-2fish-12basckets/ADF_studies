# Adversarial Defence Studies

The goal of this study is to create workable packages of existing adversarial attack and defend methods, and other visualizing tools in pytorch, and to create my own algorithm.

directories structures
```
    --raw_efforts
        README.md           contains schedules
        --paper_review      weekly paper review
            week01.md
            week02.md
            ...
        lab_notes.md        contains errorlog and research directions
    --src
        --module              common data/trainer modules
        --attacks             adversarial attack modules
        --defences            adversarial defence modules
        --special             special models
        --reproductions     directory for reproductions of previous works
    --projects
        --vs                to test existing attack vs defence
        --poke              to find ways to beat state of the art defence
        --shield            to find ways to defend state of the art attack

```

currently focusing on:  
**establishing src**
* reproduce Attention
* setup [Madry](https://paperswithcode.com/paper/towards-deep-learning-models-resistant-to), [DP](https://paperswithcode.com/paper/certified-robustness-to-adversarial-examples), [poly](https://github.com/locuslab/convex_adversarial)
* project vs

**continue reading paper**
* differential privacy
* parseval network
* outer polytope
* certified defence
* certified distributed

