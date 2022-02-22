import numpy as np

# row vector
def rv(values_list):
    return np.array([values_list])

# column vector
def cv(values_list):
    return rv(values_list).T

# Length
def length(col_v):
    return np.sqrt(np.sum(col_v * col_v))

#  Normalize a column vector
def normalize(col_v):
    return col_v/length(col_v)

# Code for signed distance
def signed_distance(x, th, th0):
    return (np.dot(th.T, x)+th0) # or (np.dot(th.T, x)+th0)/length(th)

# Code for side of hyperplane
def positive(x, th, th0):
    return np.sign(signed_distance(x, th, th0))

# Score
def score(data, labels, th, th0):
    return np.sum(np.equal(positive(data, th, th0), labels))

# Score for m separator
def score_mat(data, labels, ths, th0s):
    pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s))
    return np.sum(pos == labels, axis=1, keepdims=True)

def best_score(data, labels, ths, th0s):
    best_index = np.argmax(score_mat(data, labels, ths, th0s))
    return cv(ths[:, best_index]), th0s[:, best_index] # th0s[:, best_index: :best_index+1]

