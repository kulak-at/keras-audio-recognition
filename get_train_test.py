import numpy as np
from sklearn.model_selection import train_test_split
from convert_data import get_labels

# FIXME: fix this code
def get_train_test(split_ratio=0.6, random_state=42):
    labels, indices, _ = get_labels()

    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))
    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state)