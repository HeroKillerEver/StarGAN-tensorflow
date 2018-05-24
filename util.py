import tensorflow as tf
import numpy as np
import pickle   

def total_params(variables):
    total = 0
    for variable in variables:
        total += np.prod(variable.shape.as_list())
    return total


def preprocess():

    lines = [line.rstrip() for line in open('./data/list_attr_celeba.txt', 'r')]

    columns = lines[1].split()
    indexes = []
    labels = []
    attr2idx = {}
    for idx, attr in enumerate(columns):
        attr2idx[attr] = idx

    x2y = {}
    lines = lines[2:]
    for line in lines:
        line_ls = line.split()
        indexes.append(line_ls[0])
        labels.append([int(_ == '1') for _ in line_ls[1:]] )
        x2y[line_ls[0]] = [int(_ == '1') for _ in line_ls[1:]]

    labels = np.array(labels)

    with open('./data/attr2idx.pkl', 'wb') as f:
        pickle.dump(attr2idx, f)
    with open('./data/x2y.pkl', 'wb') as f:
        pickle.dump(x2y, f)
        
    print "Data preprocessed done......"


if __name__ == '__main__':
    preprocess()