
The Julia/Knet implementation of [<span class="underline">Semi-supervised Classification With Graph Convolutional Networks</span>](https://arxiv.org/pdf/1609.02907.pdf) [1].

## Usage

```julia train.jl```

## Parameters

      --dataset: The name of the dataset
      --model: The name of the model: gcn, gcn_cheby or dense
      --epochs: Number of epochs to train
      --lr: Initial learning rate
      --weight_decay: Weight for L2 loss on embedding matrix
      --hidden: Number of units in hidden layer
      --pdrop: Dropout rate (1 - keep probability) 
      --window_size: Tolerance for early stopping (# of epochs)
      --load_file: The path to load a saved model
      --num_of_runs: The number of randomly initialized runs 
      --save_epoch_num: The number of epochs to save the model 
      --chebyshev_max_degree: Maximum Chebyshev polynomial degree 

## Colab 
[<span class="underline">Colab link</span>](https://colab.research.google.com/drive/1yoe5yyJg-7gJ70Zp2AcIG0X1ey_XOoK3?authuser=1#scrollTo=Xd2rEUHRuFkn&uniqifier=2)

## Original Code

[<span class="underline">https://github.com/tkipf/gcn</span>](https://github.com/tkipf/gcn)

[<span class="underline">https://github.com/tkipf/pygcn</span>](https://github.com/tkipf/pygcn)


## Datasets

**1-** [<span class="underline">Citeseer</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**2-** [<span class="underline">Cora</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**3-** [<span class="underline">Pubmed</span>](https://github.com/kimiyoung/planetoid/tree/master/data)

**4-** [<span class="underline">NELL</span>](http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz)

## References

[1] Thomas N. Kipf, Max Welling. 2017, [<span class="underline">Semi-supervised Classification With Graph Convolutional Networks</span>](https://arxiv.org/pdf/1609.02907.pdf), In *International Conference on Learning Representations (ICLR)*
