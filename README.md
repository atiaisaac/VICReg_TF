# VICReg

The main issue with self-supervised models is on how to not produce constant and non-informative outputs. Although existing models have shown to prevent a collapsing solution, they do so in a way that is not fully understood. VICReg or Variance-Invariance-Covariance Regularization is a method that has shown to explicitly prevent a collapsing solution using a loss function with 3 terms. Below is an image depicting the architecture;

![vicreg](https://camo.githubusercontent.com/90f9241e251412881c9e6461140fc625384141778f25038662ac28d5a3db671f/68747470733a2f2f67656e6572616c6c79696e74656c6c6967656e742e636f6d2f7669637265672f7669637265675f6172636869746563747572652e706e67)

Just as with most self-supervised learning models, VICReg training begins with applying random augmentations to image dataset to produce two different views. The two views are fed into a joint embedding architecture with a Resnet backbone to produce embeddings. And the embeddings are then fed into a 3 layer MLP called the expander to transform them into representations for applying the loss terms.

The loss terms which consists of;
1. Variance,v(Z) - is applied separately to both branches on each embedding over a batch. The purpose of the variance term is to preserve the standard deviation of the embeddings above a threshold to force the vectors to not produce constant outputs - essentially preventing a collapsing solution. In this sense, one embedding vector will be different from the other.
2. Invariance,s(Z,Z') - is a distance metric between embeddings from the two branches. Since both branches are fed with augmented versions of the same image, they should be invariant to augmentations and it is the invariance loss term that sees to that. It is simply the mean squared distance between them.
3. Covariance,c(Z) - also applied separately to both branches. The covariance loss term is applied on pairs of embeddings over a batch and this helps to remove any form of correlation between them. Essentially it enforces decorrelation between the different dimensions of the embedding and this ensures that they are not producing the same information, therefore preventing informational collapse. 

The paper points out that, it is the variance and covariance terms that play the active role of preserving information and avoiding a degenerate solution. 

You can follow this wandb link to take a look at my experiments and the various parameters that changed along the way and how they affected performance. Training was done with TPU

# References

1. https://www.youtube.com/watch?v=XtgVYrQuIyA
2. ChatGPT (which helped me to understand covariance properly as well as an idea to structure the loss terms)
3. [VICReg paper](https://arxiv.org/pdf/2105.04906.pdf)
4. [VICReg implementation in Pytorch](https://github.com/facebookresearch/vicreg)



# Citation
```
@inproceedings{bardes2022vicreg,
  author  = {Adrien Bardes and Jean Ponce and Yann LeCun},
  title   = {VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning},
  booktitle = {ICLR},
  year    = {2022},
}
```
