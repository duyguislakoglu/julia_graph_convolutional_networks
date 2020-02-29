# TODO: Remove the unnecessary ones
using Knet: Knet, dir, minibatch, Data, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data, gpu, sigm, adam
using Statistics
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take
using Plots; default(fmt=:png,ls=:auto)

struct GraphConvolution
    w
    b
    f
end

GraphConvolution(in_features::Int,out_features::Int, f=relu) = GraphConvolution(param(out_features,in_features), param0(out_features), f)

# Forward and activation
(g::GraphConvolution)(x, adj) = g.f(spmatmul((g.w * x), adj) .+ g.b)
