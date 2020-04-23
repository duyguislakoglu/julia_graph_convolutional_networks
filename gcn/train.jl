using Pkg
for p in ["Knet", "Plots", "IterTools", "PyCall", "ArgParse"]
    if !haskey(Pkg.installed(),p)
        Pkg.add(p);
    end
end
using Knet
using DelimitedFiles
using Knet: KnetArray, accuracy, progress, minibatch, cycle, adam, xavier_uniform, progress!, @save, @load
using Plots
using ArgParse
using IterTools: ncycle, takenth, take
using Base.Iterators: flatten
using LinearAlgebra

include("utils.jl")
include("models.jl")

atype = gpu() >= 0 ? KnetArray{Float32,2} : SparseMatrixCSC{Float32,Int64}

function parse()
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table! s begin
        ("--dataset"; arg_type=AbstractString; default="cora"; help="the name of the dataset")
        ("--epochs"; arg_type=Int; default=200; help="number of epochs for training")
        ("--lr"; arg_type=Float64; default=0.01; help="learning rate")
        ("--weight_decay"; arg_type=Float64; default=5e-4; help="weight_decay")
        ("--hidden"; arg_type=Int; default=16; help="hidden")
        ("--pdrop"; arg_type=Float64; default=0.5; help="pdrop")
        ("--window_size"; arg_type=Int; default=10; help="window_size")
        ("--num_of_runs"; arg_type=Int; default=1; help="num_of_runs")
        ("--save_epoch_num"; arg_type=Int; default=200; help="save_epoch_num")
    end
    return parse_args(s)
end

args = parse()

function train_with_early_stopping(model, data, epochs, lr, window_size)
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

        if v_loss >= prev_val_loss
            early_stop_counter = early_stop_counter + 1
        else
            early_stop_counter = 0
        end
        if early_stop_counter == window_size
            flag = false
        end

        iter = iter + 1
        prev_val_loss = v_loss

        if iter%args["save_epoch_num"] == 0
            save_path = abspath(args["dataset"]*"-epoch-"*string(iter)*".jld2")
            @save save_path model
            print("The model is saved. Epoch: "* string(iter))
        end

        return flag
    end

    training = adam(model, ncycle(data, epochs), lr=lr)
    progress!(flag = task() for x in (x for (i,x) in enumerate(training)) if flag)

    return 1:iter, trnloss, valloss
end

function val_loss(g::GCN,x,y)
    output = g(x)[:, idx_val]
    nll(output, y[idx_val]) + (args["weight_decay"] * sum(g.layer1.w .* g.layer1.w))
end
function val_loss(g::GCN, d)
    mean(val_loss(g,x,y) for (x,y) in d)
end

function test_loss(g::GCN,x,y)
    output = g(x)[:, idx_test]
    nll(output, y[idx_test]) + (args["weight_decay"] * sum(g.layer1.w .* g.layer1.w))
end
function test_loss(g::GCN,d)
    mean(test_loss(g,x,y) for (x,y) in d)
end

# Load dataset
adj, features, labels, idx_train, idx_val, idx_test = load_dataset(args["dataset"])

(g::GCN)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (args["weight_decay"] * sum(g.layer1.w .* g.layer1.w))

labels_decoded = mapslices(argmax, labels ,dims=2)[:]

data =  minibatch(features, labels_decoded[:], length(labels_decoded))

global trn_acc = 0
global tst_acc = 0

function train()
    model = GCN(size(features,1),
                args["hidden"],
                size(labels,2),
                adj,
                args["pdrop"])

    iters, trnloss, vallos = train_with_early_stopping(model, data, args["epochs"], args["lr"], args["window_size"])

    output = model(features)
    curr_trn_accuracy = accuracy(output[:,idx_train], labels_decoded[idx_train])
    curr_tst_accuracy = accuracy(output[:,idx_test], labels_decoded[idx_test])

    println("Train accuracy: "* string(curr_trn_accuracy))
    println("Test accuracy: "* string(curr_tst_accuracy))

    global trn_acc =  trn_acc + curr_trn_accuracy
    global tst_acc =  tst_acc + curr_tst_accuracy
end

for i=1:args["num_of_runs"]
    println("Running... (#"*string(i)*")")
    train()
end

print("Train accuracy and test accuracy ")
if args["num_of_runs"] != 1
    println("(mean of "*string(args["num_of_runs"])*" runs): ")
end
println(trn_acc/args["num_of_runs"])
println(tst_acc/args["num_of_runs"])
