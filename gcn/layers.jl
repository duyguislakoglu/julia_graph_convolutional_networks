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
    w1
    w2
    w3
    b
    adj1
    adj2
    adj3
    f
end

# Constructor
function GCLayer2(in_features::Int, out_features::Int, adj1, adj2, adj3, f=relu)

    w1 = param(xavier_uniform(out_features,in_features))
    w2 = param(xavier_uniform(out_features,in_features))
    w3 = param(xavier_uniform(out_features,in_features))
    b = param0(out_features)
    GCLayer2(w1,w2,w3,b,adj1,adj2,adj3,f)

end

# Forward and activation
function (l::GCLayer2)(x)

    out = zeros(size(l.w1,1), size(x,2))

    mult = (l.w1 * x)
    out = out .+ (mult * l.adj1)

    mult = (l.w2 * x)
    out = out .+ (mult * l.adj2)

    mult = (l.w3 * x)
    out = out .+ (mult * l.adj3 .+ l.b)

    l.f.(out)
end

#################################################

struct GCLayer3
    w1
    w2
    w3
    w4
    b
    adj1
    adj2
    adj3
    adj4
    f
end

# Constructor
function GCLayer3(in_features::Int, out_features::Int, adj1, adj2, adj3, adj4, f=relu)
    w1 = param(xavier_uniform(out_features,in_features))
    w2 = param(xavier_uniform(out_features,in_features))
    w3 = param(xavier_uniform(out_features,in_features))
    w4 = param(xavier_uniform(out_features,in_features))
    b = param0(out_features)
    GCLayer3(w1,w2,w3,w4,b,adj1,adj2,adj3,adj4,f)
end

# Forward and activation
function (l::GCLayer3)(x)

    out = zeros(size(l.w1,1), size(x,2))

    mult = (l.w1 * x)
    out = out .+ (mult * l.adj1)

    mult = (l.w2 * x)
    out = out .+ (mult * l.adj2)

    mult = (l.w3 * x)
    out = out .+ (mult * l.adj3 .+ l.b)

    mult = (l.w4 * x)
    out = out .+ (mult * l.adj4 .+ l.b)

    l.f.(out)
end
