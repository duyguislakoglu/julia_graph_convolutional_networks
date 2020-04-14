using PyCall
using SparseArrays
const scipy_sparse_find = pyimport("scipy.sparse")["find"]

function load_dataset(dataset)
    py"""
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):

    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, np.transpose(features), labels, y_train, y_val, y_test, train_mask, val_mask, test_idx_reorder
    """

    adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_idx_reorder = py"load_data"("cora")

    (I, J, V) = scipy_sparse_find(adj)
    # Zero-indexing issue
    adj = sparse(I .+ 1, J .+ 1, V)
    adj = convert(Array{Float32,2}, adj)
    #adj = KnetArray(adj)
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
    return adj, features, labels, idx_train, idx_val, test_idx_reorder
end
