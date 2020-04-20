using Pkg
for p in ["Knet", "Plots", "IterTools", "PyCall", "ArgParse"]
    if !haskey(Pkg.installed(),p)
        Pkg.add(p);
    end
end
module train
using Knet
using DelimitedFiles
using Knet: KnetArray, accuracy, progress, minibatch, cycle, adam, xavier_uniform, progress!
using Plots
using ArgParse
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

function val_loss(g::GCN,x,y)
    output = g(x)[:, idx_val]
    nll(output, y[idx_val]) + (o[:weight_decay] * sum(g.layer1.w .* g.layer1.w))
end
function val_loss(g::GCN, d)
    mean(val_loss(g,x,y) for (x,y) in d)
end

function test_loss(g::GCN,x,y)
    output = g(x)[:, idx_test]
    nll(output, y[idx_test]) + (o[:weight_decay] * sum(g.layer1.w .* g.layer1.w))
end
function test_loss(g::GCN,d)
    mean(test_loss(g,x,y) for (x,y) in d)
end

(g::GCN)(x,y) = nll(g(x)[:, idx_train], y[idx_train]) + (o[:weight_decay] * sum(g.layer1.w .* g.layer1.w))


function main(args=ARGS)
    # Handling arguments
    s = ArgParseSettings()
    s.description="train.jl"

    s.exc_handler=ArgParse.debug_handler
    @add_arg_table! s begin
        ("--dataset"; arg_type=AbstractString; default="cora"; help="the name of the dataset")
        ("--epochs"; arg_type=Int; default=200; help="number of epochs for training")
        ("--lr"; arg_type=Float64; default=0.01; help="learning rate")
        ("--weight_decay"; arg_type=Float64; default=5e-4; help="weight_decay")
        ("--hidden"; arg_type=Int; default=16; help="hidden")
        ("--pdrop"; arg_type=Float64; default=0.5; help="pdrop")
        ("--window_size"; arg_type=Int; default=10; help="window_size")
        ("--atype"; default=(Knet.gpu()>=0 ? "KnetArray{Float32,2}" : "SparseMatrixCSC{Float32,Int64}"); help="array type: Array for cpu, KnetArray for gpu")
    end

    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o = parse_args(args, s; as_symbols=true)
    atype = eval(Meta.parse(o[:atype]))

    # Load dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset(o[:dataset])
    model = GCN(size(features,1),
                o[:hidden],
                size(labels,2),
                adj,
                o[:pdrop])

    labels_decoded = mapslices(argmax, labels ,dims=2)[:]

    data =  minibatch(features, labels_decoded[:], length(labels_decoded))

    #if
    iters, trnloss, vallos = mytrain!(model, data, o[:epochs], o[:lr], o[:window_size])
    plot(iters, [trnloss, vallos] , xlim=(1:3),labels=[:trn :val :tst], xlabel="epochs", ylabel="loss")
    #else

    #end
    png(o[:dataset])

    output = model(features)
    print(accuracy(output[:,idx_train], labels_decoded[idx_train]))
    print(accuracy(output[:,idx_test], labels_decoded[idx_test]))

end
PROGRAM_FILE=="train.jl" && main(ARGS)
end
