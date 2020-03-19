using Pkg
for p in ["Knet", "Plots", "IterTools"]
    if !haskey(Pkg.installed(),p)
        Pkg.add(p);
    end
end
using DelimitedFiles
using Knet: KnetArray, accuracy, progress, minibatch, cycle, adam, sgd
using Plots
using IterTools: ncycle, takenth, take
using Base.Iterators: flatten
using LinearAlgebra

include("utils.jl")
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

(g::GCN)(x,y) = nll(g(x)[:, idx_train], y[idx_train])

output = model(features)
labels_decoded = mapslices(argmax, labels ,dims=2)[:]
loss = nll(output[:,idx_train], labels_decoded[idx_train])
acc = accuracy(output[:,idx_train], labels_decoded[idx_train])
using AutoGrad
J = @diff model(features, labels_decoded)
gr_check = grad(J, params(model)[1])

dtrn =  minibatch(features, labels_decoded[:], length(labels_decoded))
function train_with_results(model, dtrn, epoch, lr)
    training = adam(model, ncycle(dtrn, epoch), lr=lr)
    snapshot() = model(dtrn)
    snapshots = (snapshot() for x in takenth(progress(training),length(dtrn)))
    res = collect(flatten(snapshots))
    return res
end
results = train_with_results(model, dtrn, arguments.epochs, arguments.lr)
trnloss = Array{Float32}(results)
plot(trnloss, ylim=(0.0,2.0),labels=[:trnloss],xlabel="Epochs",ylabel="Loss")
output = model(features)
accuracy(output[:,idx_train], labels_decoded[idx_train])
output = model(features)
accuracy(output[:,idx_test], labels_decoded[idx_test])