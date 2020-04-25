using Knet: relu, param, param0, dropout, mat

struct GCLayer
    w
    b
    adj
    f
end

# Constructor
GCLayer(in_features::Int, out_features::Int, adj, f=relu) = GCLayer(param(xavier_uniform(out_features,in_features)), param0(out_features), adj, f)

# Forward and activation
function (l::GCLayer)(x)
    mult = (l.w * x)
    x = mult * l.adj .+ l.b
    l.f.(x)
end

struct DenseLayer
    w
    b
    f
    p
end

# Constructor
DenseLayer(in_features::Int, out_features::Int, f=relu, pdrop=0) = DenseLayer(param(xavier_uniform(out_features,in_features)), param0(out_features), f, pdrop)

# Forward and activation
(d::DenseLayer)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)
