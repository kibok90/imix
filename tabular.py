import os
import torch
import numpy as np

class TabularDataLoader(object):

    def __init__(self, data, targets, batch_size=512, shuffle=False, drop_last=False, num_copies=1):
        assert (data.shape[0] == targets.shape[0])

        self.num_data = data.shape[0]
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_copies = num_copies
        self.num_batches = self.num_data // batch_size

        last = self.num_data - self.num_batches * batch_size
        if (not drop_last) and (last > 0):
            self.num_batches += 1
            self.num_data_iter = self.num_data
        else:
            self.num_data_iter = self.num_data - last

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            randind = torch.randperm(self.num_data)
            self.data = self.data[randind]
            self.targets = self.targets[randind]
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_data_iter:
            raise StopIteration
        data = self.data[self.index:(self.index+self.batch_size)]
        targets = self.targets[self.index:(self.index+self.batch_size)]
        self.index += len(data)

        if self.num_copies > 1:
            data = [data] * self.num_copies

        return data, targets


def covtype(root, channels=[10, 4, 40], do_normalize=True):
    if os.path.exists(os.path.join(root, 'covtype.npy')):
        mat = np.load(os.path.join(root, 'covtype.npy'), allow_pickle=True)
    else:
        mat = np.genfromtxt(os.path.join(root, 'covtype.data.gz'), dtype=np.float32, delimiter=',')
        mat[:,-1] -= 1
        np.save(os.path.join(root, 'covtype.npy'), mat)

    data, targets = mat[:,:-1], mat[:,-1].astype(np.int32)

    # channels: 10 continuous, 4 binary, 40 binary
    if do_normalize:
        if isinstance(channels, (tuple, list)):
            data_cont = data[:, :channels[0]]
            data[:, :channels[0]] = (data_cont - data_cont.mean(axis=0)) / data_cont.std(axis=0)
        else:
            data = (data - data.mean(axis=0)) / data.std(axis=0)

    trainval = 11340 + 3780
    data, targets = torch.tensor(data), torch.tensor(targets)
    return (data[:trainval], targets[:trainval]), (data[trainval:], targets[trainval:])


def higgs(root, mode='100k', channels=21, do_normalize=True):
    if os.path.exists(os.path.join(root, 'HIGGS.npy')):
        mat = np.load(os.path.join(root, 'HIGGS.npy'), allow_pickle=True)
    else:
        mat = np.genfromtxt(os.path.join(root, 'HIGGS.csv.gz'), dtype=np.float32, delimiter=',')
        np.save(os.path.join(root, 'HIGGS.npy'), mat)

    data, targets = mat[:, 1:(channels+1)].astype(np.float32), mat[:, 0].astype(np.int32)

    if do_normalize:
        if isinstance(channels, (tuple, list)):
            data_cont = data[:, :channels[0]]
            data[:, :channels[0]] = (data_cont - data_cont.mean(axis=0)) / (data_cont.std(axis=0) + 1e-7)
        else:
            data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-7)

    test = 500000
    data, targets = torch.tensor(data), torch.tensor(targets)
    if mode.lower() == '100k':
        np.random.seed(1)
        trainval = np.random.choice(len(data)-test, 100000, replace=False)
        return (data[trainval], targets[trainval]), (data[-test:], targets[-test:]) 
    else:
        return (data[:-test], targets[:-test]), (data[-test:], targets[-test:]) 

