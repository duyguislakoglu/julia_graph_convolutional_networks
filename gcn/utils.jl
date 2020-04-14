using PyCall
using SparseArrays
const scipy_sparse_find = pyimport("scipy.sparse")["find"]

function load()
# TODO: Change for multiple datasets
    # PyCall
    py"""
import numpy as np
import scipy.sparse as sp

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(path="cora/", dataset="cora"):
    #Load citation network dataset (cora only for now)
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    return adj, np.transpose(features), labels
    """

    adj, features, labels = py"load_data"()

    (I, J, V) = scipy_sparse_find(adj)
    # Zero-indexing issue
    adj = sparse(I .+ 1, J .+ 1, V)
    adj = convert(Array{Float32,2}, adj)
   # adj = KnetArray(adj)
    #adj = Array(adj)

    (I, J, V) = scipy_sparse_find(features)
    # Zero-indexing issue
    features = sparse(I .+ 1, J .+ 1, V)
    features = convert(Array{Float32,2}, features)
    #features = KnetArray(features)
    #features = Array(features)

    # TODO: Uncomment the following
    # Normalize feature
    # features = features./sum(features,2)
    # Add identity matrix
    #adj += sparse(I, size(adj,1), size(adj,2))
    # Normalize
    #adj = adj./sum(adj,2)

    #adj = convert(KnetArray{Float32,2}, adj)
    #features = convert(KnetArray{Float32,2}, features)
    #labels = convert(KnetArray{Float32,2}, labels)

	idx_train = 1:140
    idx_val = 200:499
    #idx_test = 500:size(features,2)
    idx_test = 500:1499
    return adj, features, labels, idx_train, idx_val, idx_test
end
