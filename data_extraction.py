import numpy as np
from scipy.io import loadmat

def get_X_y(file_name, encoding='binary', normalize=False, k_target=0, n=0):
    """
    Args:
        file_name [string]: path of file from current directory
        encoding [string]: 'binary' or 'value'
        normalize [boolean]: whether to normalize the data using buffer identity functions (NOTE: ONLY active for
            encoding = 'binary')
        k_target [int]: target k value for normalization
        n [int]: length of input bit string
    Returns:
        X [np.array]: fature matrix of shape (n_examples x d)
        y [np.array]: label vector of shape (n_sampels,)
    """
    datafile = loadmat(file_name)
    POS_dataset = NEG_dataset = None
    if encoding == 'binary':
        POS_dataset = datafile['POS_dataset']
        NEG_dataset = datafile['NEG_dataset']
        if normalize:
            POS_dataset, NEG_dataset = normalize_dimension(POS_dataset,NEG_dataset,n,k_target)
    elif encoding == 'value':
        POS_dataset = datafile['POS_storage']
        NEG_dataset = datafile['NEG_storage']
    X = np.concatenate((POS_dataset, NEG_dataset))
    y = np.concatenate((np.ones(POS_dataset.shape[0], ).astype(int), np.zeros(NEG_dataset.shape[0], )))
    return X, y

def get_training_test_data(X, y, n_train_total, n_test_total=-1, random_state=2508):
    """
    Obtain a training and test set with the desired number of examples from the dataset
    Args:
        X: np.array, parent design matrix of size (n_samples * d)
        y: np.array, parent label vector of size (n_samples,)
        n_train_total: int, number of training points
        n_test_total: int, number of test points (Default: -1, all points not chosen for
            training are included)
        random_state: int, seed the random number genenator for reproducability
    Returns:
        X_train: np.array, training data of size (n_train_total * d)
        y_train: np.array, training label of size (n_train_total,)
        X_test: np.array, test data of size (n_test_total * d)
        y_test: np.array, test label of size (n_test_total * d)
    """
    X_train, y_train, X_test, y_test = get_subset_n(X,y,n_train_total,random_state)
    if n_test_total != -1:
        X_test, y_test, _, _ = get_subset_n(X_test, y_test, n_test_total,random_state)
    return X_train, y_train, X_test, y_test

def get_subset_n(X, y, n_total_points, random_state=2508):
    """
    Randomly extract a balanced dataset of size n_total_points from a parent dataset
    Args:
        X: np.array, parent design matrix of size (n_samples * d)
        y: np.array, parent label vector of size (n_samples,)
        n_total_points: int, total points to extract (including both classes)
            requires: (n_total_points / 2) < n_pos_samples, n_neg_samples
        random_state: int, seed the random number genenator for reproducability

    Returns:
        X_chosen: np.array, balanced subset design matrix of size (n_total_points * d)
        y_chosen: np.array, subset label vector of size (n_total_points,)
        X_left: np.array, leftover design matrix of size ((n_samples - n_total_points) * d)
        y_left: np.array, leftover label vector of size ((n_samples - n_total_points),)
    """
    n_class_examples = int(n_total_points / 2)
    np.random.seed(random_state)
    class0_ind = np.random.choice(np.where(y == 0)[0], n_class_examples, replace=False)
    class1_ind = np.random.choice(np.where(y == 1)[0], n_class_examples, replace=False)
    chosen_ind = np.concatenate((class0_ind, class1_ind))
    X_chosen = np.concatenate((X[class1_ind], X[class0_ind]))
    y_chosen = np.concatenate((np.ones(n_class_examples, ).astype(int), np.zeros(n_class_examples, ).astype(int)))
    X_left = np.delete(X,chosen_ind,axis=0)
    y_left = np.delete(y,chosen_ind,axis=0)
    return X_chosen, y_chosen, X_left, y_left

def normalize_dimension(POS_dataset, NEG_dataset, n, k_target):
    """Buffer a dataset with low k to one with higher k with identity functions """
    k_add = int(k_target - np.shape(POS_dataset)[1] / n)
    if k_add == 0:
        return POS_dataset, NEG_dataset
    POS_dataset_ = np.c_[POS_dataset, np.zeros((np.shape(POS_dataset)[0],k_add * n))]
    NEG_dataset_ = np.c_[NEG_dataset, np.zeros((np.shape(NEG_dataset)[0],k_add * n))]
    return POS_dataset_.astype(int), NEG_dataset_.astype(int)