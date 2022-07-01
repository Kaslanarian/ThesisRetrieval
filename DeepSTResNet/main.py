import argparse
import torch
import os
import time
import h5py
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    sns.set()
except:
    pass

from data import load_data, DATAPATH
from model import STresnet


def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(np.array(f['num']))
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(np.array(f['X_train_%i' % i]))
        X_test.append(np.array(f['X_test_%i' % i]))
    Y_train = np.array(f['Y_train'])
    Y_test = np.array(f['Y_test'])
    external_dim = np.array(f['external_dim'])
    timestamp_train = np.array(f['T_train'])
    timestamp_test = np.array(f['T_test'])
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim,
          timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--cache",
                    type=bool,
                    help='use cache to load/save data (default True).',
                    default=True)
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help='random seed (default: 42).')
parser.add_argument('--test-size',
                    type=float,
                    default=0.1,
                    help='test dataset size (default 0.1).')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr',
                    type=float,
                    default=0.002,
                    help='learning rate (default: 0.002)')
parser.add_argument('--bn',
                    type=bool,
                    default=False,
                    help='whether use BN in model (default False)')
parser.add_argument('--n-residual',
                    type=int,
                    default=3,
                    help='number of residual unit (default 3)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='how many batches to wait before logging training status')

args = parser.parse_args()
len_closeness = 3
len_period = 1
len_trend = 1
use_cuda = not args.no_cuda and torch.cuda.is_available()

ts = time.time()
fname = os.path.join(
    DATAPATH, 'CACHE', 'TaxiBJ_C{}_P{}_T{}.h5'.format(len_closeness,
                                                      len_period, len_trend))
if os.path.exists(fname) and args.cache:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
        fname)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        test_size=args.test_size,
        preprocess_name='preprocessing.pkl',
        meta_data=True,
        meteorol_data=True,
        holiday_data=True)
    if args.cache:
        cache(fname, X_train, Y_train, X_test, Y_test, external_dim,
              timestamp_train, timestamp_test)
t = time.time() - ts
print("Data process time : {:.4f}s".format(t))


class STDataset(torch.utils.data.Dataset):
    """Some Information about STDataset"""
    def __init__(self, Xc, Xp, Xt, Ext, y):
        super(STDataset, self).__init__()
        self.Xc = Xc
        self.Xp = Xp
        self.Xt = Xt
        self.Ext = Ext
        self.y = y

    def __getitem__(self, index):
        return (
            self.Xc[index],
            self.Xp[index],
            self.Xt[index],
            self.Ext[index],
        ), self.y[index]

    def __len__(self):
        return len(self.Xc)


torch.random.manual_seed(args.seed)

train_dataset = STDataset(
    *(torch.tensor(x).float() for x in X_train),
    torch.tensor(Y_train).float(),
)
test_dataset = STDataset(
    *(torch.tensor(x).float() for x in X_test),
    torch.tensor(Y_test).float(),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=True,
)

device = torch.device(
    "cuda" if use_cuda and torch.cuda.is_available() else "cpu")

net = STresnet(
    bn=args.bn,
    nb_residual_unit=args.n_residual,
).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()
rmse_list = []

print("Start training")
for epoch in range(1, args.epochs + 1):
    net.train()

    for batch_X, batch_y in train_loader:
        batch_X = [x.to(device) for x in batch_X]
        batch_y = batch_y.to(device)
        output = net.forward(*batch_X)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    sum_loss = 0.
    for batch_X, batch_y in test_loader:
        batch_X = [x.to(device) for x in batch_X]
        batch_y = batch_y.to(device)
        output = net.forward(*batch_X)
        loss = criterion(output, batch_y)
        sum_loss += loss.item() * len(batch_y)

    final_loss = (sum_loss / len(test_dataset))**0.5 * mmn._max / 2
    print("Epoch {}, test RMSE {:.4f}".format(epoch, final_loss))
    rmse_list.append(final_loss)

plt.plot(range(1, args.epochs + 1), rmse_list, label='test RMSE')
plt.legend()
plt.savefig("src/train_baseline.png")
