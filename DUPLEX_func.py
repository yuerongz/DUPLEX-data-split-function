import numpy as np
import scipy.spatial.distance as dist


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
    train_idxs = []
    test_idxs = []

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
    remaining_idxs = list(range(dist_m.shape[0]))
    r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
    dist_m[r1, :] = 0
    dist_m[r2, :] = 0  # remove distances along deleting item rows
    train_idxs.extend([r1, r2])
    remaining_idxs.remove(r1)
    remaining_idxs.remove(r2)
    # allocate the first two data sets to testing set
    dist_m2 = dist_m[:, [r1, r2]].copy()
    dist_m[:, r1] = 0
    dist_m[:, r2] = 0
    r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
    test_idxs.extend([r1, r2])
    remaining_idxs.remove(r1)
    remaining_idxs.remove(r2)
    dist_m[:, train_idxs] = dist_m2
    dist_m[dist_m == 0] = np.nan

    # while loop, fill test set
    while len(test_idxs) < test_set_size:
        # 1 samples to Train
        r_of_remains = np.argmax(np.nanmin(dist_m[:, train_idxs][remaining_idxs], axis=0))
        r = remaining_idxs[int(r_of_remains)]
        train_idxs.append(r)
        remaining_idxs.remove(r)

        # 1 samples to Test
        r_of_remains = np.argmax(np.nanmin(dist_m[:, test_idxs][remaining_idxs], axis=0))
        r = remaining_idxs[int(r_of_remains)]
        test_idxs.append(r)
        remaining_idxs.remove(r)

    # put the rest of the data into train set
    if len(remaining_idxs) > 0:
        train_idxs.extend(remaining_idxs)

    assert len(train_idxs) == train_set_size, "Duplex: train set size not matching expectation!"
    assert len(test_idxs) == test_set_size, "Duplex: test set size not matching expectation!"

    # train = size(batch, 192, 15), test = size(batch, 1, 1)
    train_data = all_data[train_idxs]
    test_data = all_data[test_idxs]
    return train_data, test_data

