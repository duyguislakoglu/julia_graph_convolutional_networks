using Knet: Adam

# TODO: Change load name
from graph_convolutional_networks.utils import load
from graph_convolutional_networks.models import GCN

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
args = args(200, 0.01, 5e-4, 16, 0.5)

# Model and optimizer
model = GCN(nfeat=size(features,2),
            nhid = args.hidden,
            # TODO: Change nclass
            nclass = size(labels,2)
            #nclass=max(labels) + 1,
            pdrop = args.pdrop)

dtrn = features[idx_train]
atrn = adj[idx_train]
dval = features[idx_val]
aval = adj[idx_val]
dtst = features[idx_test]
atst = adj[idx_test]

function train_results(model, dtrn, dval, epoch, lr)
    training = adam(model, ncycle(dtrn, epoch), lr=lr)
    # TODO: Our model does not take one argument, rewrite zeroone
    snapshot() = (deepcopy(model), model(dtrn), model(dval), 1 - zeroone(model,dtrn), 1 - zeroone(model,eval))
    # TODO: Why length(dtrn)
    snapshots = (snapshot() for x in takenth(progress(training),length(dtrn)))
    lin = reshape(collect(flatten(snapshots)),(5,:))
    Knet.save(file,"results",lin)
    return lin
end

function test()
    # TODO: Our model does not take one argument, rewrite zeroone
    snapshot() = (deepcopy(model), model(dtst), 1 - zeroone(model, dtst))
end

train_results(model, dtrn, dval, args.epoch, args.lr)
test()
