using Knet: dropout, nll, Data
using Statistics

# TODO: Can convert to chain structure
struct GCN
    layer1::GraphConvolution
    layer2::GraphConvolution
    pdrop
end

GCN(nfeat::Int, nhid::Int, nclass::Int, pdrop=0) = GCN(layer1(nfeat, nhid), layer2(nhid, nclass, identity), pdrop)

function (c::GCN)(x, adj)
    x = layer1(x, adj)
    x = dropout(x, c.pdrop)
    layer2(x, adj)
end

(for l in c.layers; x = l(x); end; x)
(c::GCN)(x,y) = nll(c(x),y)
(c::GCN)(d::Data) = mean(c(x,y) for (x,y) in d)
