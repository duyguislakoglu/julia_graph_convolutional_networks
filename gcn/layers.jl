using Knet: relu, param, param0

struct GCLayer
    w
    b
    adj
    f
end

GCLayer(in_features::Int, out_features::Int, adj, f=relu) = GCLayer(param(out_features,in_features), param0(out_features), adj, f)

# Forward and activation
function (l::GCLayer)(x)
    x = mult = (l.w * x)
    x = mult * l.adj .+ l.b
    l.f.(x)
end