Thomas N. Kipf, Max Welling. 2017, [<span class="underline">Semi-supervised Classification With Graph Convolutional Networks</span>](https://arxiv.org/pdf/1609.02907.pdf), In *International Conference on Learning Representations (ICLR)*

## Requirements

  * KNET

## Usage

```julia train.jl```


**Original Code:**

[<span class="underline">https://github.com/tkipf/gcn</span>](https://github.com/tkipf/gcn)

[<span class="underline">https://github.com/tkipf/pygcn</span>](https://github.com/tkipf/pygcn)


**Datasets:**![](/media/image1.png)

**1-** [<span class="underline">Citeseer</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**2-** [<span class="underline">Cora</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**3-** [<span class="underline">Pubmed</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**4-** [<span class="underline">NELL</span>](http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz)

**Baselines**

**1- Semi-Supervised Node Classification**

1.  > Label propagation (LP) (Zhu et al., 2003)

2.  > Semi-supervised embedding (SemiEmb) (Weston et al., 2012)

3.  > Manifold regularization (ManiReg) (Belkin et al., 2006)

4.  > Skip-gram based graph embeddings (DeepWalk) (Perozzi et al., 2014)

5.  > Planetoid (Yang et al., 2016)

6.  > Iterative classification algorithm (ICA) (Lu & Getoor, 2003)
