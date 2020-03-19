using Pkg
for p in ["Knet", "Plots", "IterTools"]
    if !haskey(Pkg.installed(),p)
        Pkg.add(p);
    end
end
using DelimitedFiles
using Knet: KnetArray, accuracy, progress, minibatch, cycle, adam, sgd
using IterTools: ncycle, takenth, take
using Base.Iterators: flatten

include("utils.jl")
include("layers.jl")
include("models.jl")

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load()

# TODO: take user inputs
struct args
    epochs
    lr
    weight_decay
    hidden
    pdrop
end

arguments = args(200, 0.01, 5e-4, 16, 0.5)

# Model and optimizer
model = GCN(size(features,1),
            arguments.hidden,
            size(labels,2),
            adj,
            arguments.pdrop)

output = model(features)
labels_decoded = mapslices(argmax, labels ,dims=2)
nll(output[:,idx_train], labels_decoded[idx_train])
accuracy(output[:,idx_train], labels_decoded[idx_train])