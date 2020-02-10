import numpy as np
import scipy.spatial.distance as dist
from torch.utils.data import TensorDataset
from LSTM_DUPLEX_numba import allocating_items


def duplex_data_split(all_data, train_test_ratio):
    """
    Apply DUPLEX split for the provided object.
    all_data = torch.utils.data.TensorDataset, contains tuple(inputs, outputs), where:
        inputs = torch.tensor, size = (batch, sequence_len=192, input_dim=15)
        outputs = torch.tensor, size = (batch, 1, output_dim=1)
        originally prepared for LSTM network training tasks.
    Output: TensorDataset_train, TensorDataset_test
    """
    assert train_test_ratio >= 1, "Duplex split: train-test ratio needs to be no less than one!"

    inputs, outputs = all_data[:]
    train_set_size = round(inputs.size(0) * train_test_ratio / (train_test_ratio + 1))
    test_set_size = int(inputs.size(0) - train_set_size)
    train_idxs = -np.ones(train_set_size, dtype=int)
    test_idxs = -np.ones(test_set_size, dtype=int)

    # process the inputs(3D) and outputs(3D) from the TensorDataset into 2-D matrices
    # inputs: took only the Q and SeaLev sequences, calculate the max() and min() for both sequences,
    #           use them as inputs indicators to avoid massive calculation efforts
    inputs = np.reshape(inputs.numpy()[:, :, [0, -1]], (inputs.size(0), inputs.size(1) * 2))     # (batch, 192, 15)
    inputs = np.array([[max(curr[:192]), min(curr[:192]), max(curr[192:]), min(curr[192:])] for curr in inputs])
    outputs = np.reshape(outputs.numpy(), (outputs.size(0), 1))   # (batch, 1, 1)
    # arrange inputs and outputs into one matrix by join columns
    flattened_all_data = np.concatenate((inputs, outputs), axis=1)   # format=np.ndarray()

    # allocate the first two data sets to training set
    dist_m = dist.cdist(flattened_all_data, flattened_all_data, 'sqeuclidean')
    remaining_idxs = np.arange(dist_m.shape[0])
    r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
    dist_m[r1, :] = 0
    dist_m[r2, :] = 0  # remove distances along deleting item rows
    train_idxs[[0, 1]] = [r1, r2]
    remaining_idxs[[r1, r2]] = -1
    # allocate the first two data sets to testing set
    dist_m2 = dist_m[:, [r1, r2]].copy()
    dist_m[:, r1] = 0
    dist_m[:, r2] = 0
    r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
    test_idxs[[0, 1]] = [r1, r2]
    remaining_idxs[[r1, r2]] = -1
    dist_m[:, train_idxs[:2]] = dist_m2
    dist_m[dist_m == 0] = dist_m.max()

    # while loop, fill test set
    for i in range(test_set_size-2):
        train_idxs, test_idxs, remaining_idxs = allocating_items(dist_m, train_idxs, test_idxs, remaining_idxs, i)
        print(i)

    # put the rest of the data into train set
    if len(remaining_idxs) > 0:
        train_idxs[test_set_size:] = remaining_idxs[remaining_idxs != -1]

    assert np.min(train_idxs) > -1, "Duplex: train set size not matching expectation!"
    assert np.min(test_idxs) > -1, "Duplex: test set size not matching expectation!"

    # train = size(batch, 192, 15), test = size(batch, 1, 1)
    train_data = TensorDataset(all_data[train_idxs][0], all_data[train_idxs][1])
    test_data = TensorDataset(all_data[test_idxs][0], all_data[test_idxs][1])
    return train_data, test_data

