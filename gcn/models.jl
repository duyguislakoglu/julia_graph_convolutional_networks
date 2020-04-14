using Knet: dropout, nll, Data
using Statistics
include("layers.jl")
idx_train=1:140
 
# TODO: Can convert to chain structure
struct GCN
    layer1::GCLayer
    layer2::GCLayer
    pdrop
end

GCN(nfeat::Int, nhid::Int, nclass::Int, adj, pdrop=0) = GCN(GCLayer(nfeat, nhid, adj), GCLayer(nhid, nclass, adj, identity), pdrop)

function (g::GCN)(x)
    x = g.layer1(x)
    x = dropout(x, g.pdrop)
    g.layer2(x)
end   

(g::GCN)(d::Data) = mean(g(x,y) for (x,y) in d)