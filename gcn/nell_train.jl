using Pkg
for p in ["Knet", "Plots", "IterTools","PyCall"]
    if !haskey(Pkg.installed(),p)
        Pkg.add(p);
    end
end
using DelimitedFiles
using Knet: KnetArray, accuracy, progress, minibatch, cycle, adam, xavier_uniform, progress!
using Plots
using IterTools: ncycle, takenth, take
using Base.Iterators: flatten
using LinearAlgebra

include("utils.jl")
include("models.jl")

function mytrain!(model, data, epochs, lr, window_size)
    early_stop_counter = 0
    prev_val_loss = 0
    iter = 0

    trnloss = []
    valloss = []

    flag = true

    function task()

        append!(trnloss, model(data))
        v_loss = val_loss(model, data)
        append!(valloss, v_loss)

        if v_loss == prev_val_loss
            early_stop_counter = early_stop_counter + 1
        else
            early_stop_counter = 0
        end
        if early_stop_counter == window_size
            flag = false
        end
        iter = iter + 1
        prev_val_loss = v_loss
        return flag

    end

    training = adam(model, ncycle(data, epochs), lr=lr)
    progress!(flag = task() for x in (x for (i,x) in enumerate(training)) if flag)
    return 1:iter, trnloss, valloss
end

# TODO: take user inputs
struct args
    epochs
    lr
    weight_decay
    hidden
    pdrop
    window_size
end

arguments = arg(200, 0.01, 1e-5, 64, 0.1, 10)

function val_loss(g::GCN,x,y)
    output = g(x)[:, idx_val]
    nll(output, y[idx_val]) + (arguments.weight_decay * sum(g.layer1.w .* g.layer1.w))
end
function val_loss(g::GCN, d)
    mean(val_loss(g,x,y) for (x,y) in d)
end

function test_loss(g::GCN,x,y)
    output = g(x)[:, idx_test]
    nll(output, y[idx_test]) + (arguments.weight_decay * sum(g.layer1.w .* g.layer1.w))
end

function test_loss(g::GCN,d)
    mean(test_loss(g,x,y) for (x,y) in d)
end
(g::GCN)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (arguments.weight_decay * sum(g.layer1.w .* g.layer1.w))

adj, features, labels, idx_train, idx_val, idx_test = load_dataset("nell")

model = GCN(size(features,1),
            arguments.hidden,
            size(labels,2),
            adj,
            arguments.pdrop)

labels_decoded = mapslices(argmax, labels ,dims=2)[:]

data =  minibatch(features, labels_decoded[:], length(labels_decoded))
function train_with_results(model, data, epoch, lr)
    training = adam(model, ncycle(data, epoch), lr=lr)

    function snapshot()
        out = model(features)
        model(data), val_loss(model, data)
    end
    snapshots = (snapshot() for x in takenth(progress(training),length(data)))
    res = reshape(collect(flatten(snapshots)),(2,:))
    return res
end

results = train_with_results(model, data, 200, arguments.lr)
trnloss, valloss = Array{Float32}(results[1,:]), Array{Float32}(results[2,:])

plot([trnloss, valloss], ylim=(0.0,100),labels=[:trnloss :valloss],xlabel="Epochs",ylabel="Loss")

png("nell")

output = model(features)
print(accuracy(output[:,idx_train], labels_decoded[idx_train]))
print(accuracy(output[:,idx_test], labels_decoded[idx_test]))
