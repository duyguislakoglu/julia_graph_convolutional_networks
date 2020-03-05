# TODO: Remove the unnecessary ones
using Knet: Knet, dir, minibatch, Data, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data, gpu, sigm, adam
using Statistics
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take
using Plots; default(fmt=:png,ls=:auto)
using SparseArrays: spmatmul
const scipy_sparse_find = pyimport("scipy.sparse")["find"]

struct GraphConvolution
    w
    b
    f
end

GraphConvolution(in_features::Int,out_features::Int, f=relu) = GraphConvolution(param(out_features,in_features), param0(out_features), f)

# Forward and activation
# TODO: Handle adj. dimension match
function (g::GraphConvolution)(x, adj)
    #print(size(g.w))
    #print(size(x))

    mult = (g.w * x)

    (I, J, V) = scipy_sparse_find(mult)
    mult = sparse(I .+ 1, J .+ 1, V)
    mult = convert(SparseMatrixCSC{Float64,Int64}, mult) 
 
    print(size((spmatmul(mult, adj) .+ g.b)))
    g.f.(spmatmul(mult, adj) .+ g.b)
    
end