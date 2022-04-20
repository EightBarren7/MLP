import torch.utils.data as Data
import pandas as pd


def dataloader(batch_size, shuffle=True):
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # X = digits.data
    # y = digits.target
    # Y = []
    # for i in y:
    #     Y.append([i])

    train_data = pd.read_csv('mnist_train.csv').values
    test_data = pd.read_csv('mnist_test.csv').values
    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]
    test_X = test_data[:, 1:]
    test_y = test_data[:, 0]
    len_train_data = len(train_X)
    len_test_data = len(test_X)
    train_dataset = []
    test_dataset = []
    for i in range(len(train_X)):
        train_dataset.append((train_X[i], train_y[i]))
    for i in range(len(test_X)):
        test_dataset.append((test_X[i], test_y[i]))
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_iter = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_iter, test_iter, len_train_data, len_test_data

