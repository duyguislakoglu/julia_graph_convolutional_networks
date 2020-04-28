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
        ("--dataset"; arg_type=AbstractString; default="cora"; help="The name of the dataset.")
        ("--model"; arg_type=AbstractString; default="gcn"; help="The name of the model: gcn, gcn_cheby or dense")
        ("--epochs"; arg_type=Int; default=200; help="Number of epochs to train.")
        ("--lr"; arg_type=Float64; default=0.01; help="Initial learning rate.")
        ("--weight_decay"; arg_type=Float64; default=5e-4; help="Weight for L2 loss on embedding matrix.")
        ("--hidden"; arg_type=Int; default=16; help="Number of units in hidden layer.")
        ("--pdrop"; arg_type=Float64; default=0.5; help="Dropout rate (1 - keep probability).")
        ("--window_size"; arg_type=Int; default=10; help="Tolerance for early stopping (# of epochs).'")
        ("--load_file"; default=""; help="The path to load a saved model.")
        ("--num_of_runs"; arg_type=Int; default=1; help="The number of randomly initialized runs.")
        ("--save_epoch_num"; arg_type=Int; default=250; help="The number of epochs to save the model.")
        ("--chebyshev_max_degree"; arg_type=Int; default=0; help="Maximum Chebyshev polynomial degree.")
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
            print("\nThe model is saved. Epoch: "* string(iter))
        end

        return flag
    end

    training = adam(model, ncycle(data, epochs), lr=lr)
    progress!(flag = task() for x in (x for (i,x) in enumerate(training)) if flag)

    return 1:iter, trnloss, valloss
end

function val_loss(model,x,y)
    output = model(x)[:, idx_val]
    nll(output, y[idx_val]) + (args["weight_decay"] * sum(model.layer1.w .* model.layer1.w))
end

function test_loss(model,x,y)
    output = model(x)[:, idx_test]
    nll(output, y[idx_test]) + (args["weight_decay"] * sum(model.layer1.w .* model.layer1.w))
end

##################################

if args["chebyshev_max_degree"]  ==  2

    function val_loss(model,x,y)
        output = model(x)[:, idx_val]
        nll(output, y[idx_val]) + (args["weight_decay"] * sum(model.layer1.w1 .* model.layer1.w1)) + (args["weight_decay"] * sum(model.layer1.w2 .* model.layer1.w2)) + (args["weight_decay"] * sum(model.layer1.w3 .* model.layer1.w3))
    end

    function test_loss(model,x,y)
        output = model(x)[:, idx_test]
        nll(output, y[idx_test]) + (args["weight_decay"] * sum(model.layer1.w1 .* model.layer1.w1)) + (args["weight_decay"] * sum(model.layer1.w2 .* model.layer1.w2)) + (args["weight_decay"] * sum(model.layer1.w3 .* model.layer1.w3))
            + (args["weight_decay"] * sum(model.layer1.w4 .* model.layer1.w4))
    end

elseif args["chebyshev_max_degree"]  ==  3

    function val_loss(model,x,y)
        output = model(x)[:, idx_val]
        nll(output, y[idx_val])
    end

    function test_loss(model,x,y)
        output = model(x)[:, idx_test]
        nll(output, y[idx_test])
    end
end

##################################

function val_loss(model, d)
    mean(val_loss(model,x,y) for (x,y) in d)
end
function test_loss(model,d)
    mean(test_loss(model,x,y) for (x,y) in d)
end

# Load dataset
if args["chebyshev_max_degree"] == 0
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset(args["dataset"], args["chebyshev_max_degree"])
elseif args["chebyshev_max_degree"] == 2
    adj1, adj2, adj3, features, labels, idx_train, idx_val, idx_test = load_dataset(args["dataset"], args["chebyshev_max_degree"])
elseif args["chebyshev_max_degree"] == 3
    adj1, adj2, adj3, adj4, features, labels, idx_train, idx_val, idx_test = load_dataset(args["dataset"], args["chebyshev_max_degree"])
end

if atype == KnetArray{Float32,2}
    if args["chebyshev_max_degree"]==0
        global adj = convert(atype, adj)
    else
        for i=1:length(adj)
            adj[i] = convert(atype, adj[i])
        end
    end
end
features = convert(atype, features)


(m::MLP)(x,y) = nll(m(x)[:, idx_train], y[idx_train]) + (args["weight_decay"] * sum(m.layer1.w .* m.layer1.w))
(g::GCN)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (args["weight_decay"] * sum(g.layer1.w .* g.layer1.w))
(g::GCN2)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (args["weight_decay"] * sum(g.layer1.w1 .* g.layer1.w1)) + (args["weight_decay"] * sum(g.layer1.w2 .* g.layer1.w2)) + (args["weight_decay"] * sum(g.layer1.w3 .* g.layer1.w3))
(g::GCN3)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (args["weight_decay"] * sum(g.layer1.w1 .* g.layer1.w1)) + (args["weight_decay"] * sum(g.layer1.w2 .* g.layer1.w2)) + (args["weight_decay"] * sum(g.layer1.w3 .* g.layer1.w3)) + (args["weight_decay"] * sum(g.layer1.w4 .* g.layer1.w4))

labels_decoded = mapslices(argmax, labels ,dims=2)[:]

data =  minibatch(features, labels_decoded[:], length(labels_decoded))

global trn_acc = 0
global tst_acc = 0

function train()
    if  args["model"] == "mlp"
        model = MLP(size(features,1),
                    args["hidden"],
                    size(labels,2),
                    args["pdrop"])
    elseif  args["chebyshev_max_degree"] == 2
        model = GCN2(size(features,1),
                    args["hidden"],
                    size(labels,2),
                    adj1,
                    adj2,
                    adj3,
                    args["pdrop"])
    elseif args["chebyshev_max_degree"] == 3
        model = GCN3(size(features,1),
                    args["hidden"],
                    size(labels,2),
                    adj1,
                    adj2,
                    adj3,
                    adj4,
                    args["pdrop"])
    else
        model = GCN(size(features,1),
                    args["hidden"],
                    size(labels,2),
                    adj,
                    args["pdrop"])
    end

    args["load_file"] != "" && @load args["load_file"] model

    iters, trnloss, valloss = train_with_early_stopping(model, data, args["epochs"], args["lr"], args["window_size"])

    if args["num_of_runs"] == 1
        plot(iters, [trnloss, valloss],labels=[:trn :val], xlabel="epochs", ylabel="loss")
        png(args["dataset"])
    end

    output = model(features)
    curr_trn_accuracy = accuracy(output[:,idx_train], labels_decoded[idx_train])
    curr_tst_accuracy = accuracy(output[:,idx_test], labels_decoded[idx_test])

    println("Train accuracy: " * string(curr_trn_accuracy))
    println("Test accuracy: " * string(curr_tst_accuracy))

    global trn_acc =  trn_acc + curr_trn_accuracy
    global tst_acc =  tst_acc + curr_tst_accuracy
end

for i=1:args["num_of_runs"]
    println("Running... (#"*string(i)*")")
    train()
end

println("Train accuracy and test accuracy ")
if args["num_of_runs"] != 1
    println("(mean of " * string(args["num_of_runs"])*" runs): ")
end

println(trn_acc/args["num_of_runs"])
println(tst_acc/args["num_of_runs"])
