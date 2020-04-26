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

#################################################

struct GCLayer2
    weight_list
    b_list
    support
    f
end

# Constructor
function GCLayer2(in_features::Int, out_features::Int, support, f=relu)
    weight_list = []
    b_list = []

    for i=1:length(support)
        append!(weight_list, param(xavier_uniform(out_features,in_features)))
        append!(b_list, param0(out_features))
    end
    GCLayer2(weight_list, b_list, support, f)
end

# Forward and activation
function (l::GCLayer2)(x)
    out = zeros(size(l.weight_list[1],1), size(x,2))
    for i=1:length(l.support)
        mult = (l.weight_list[i] * x)
        out = out .+ (mult * l.support .+ l.b_list[i])
    end
    l.f.(out)
end
