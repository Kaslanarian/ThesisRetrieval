import scipy.io as sio
import numpy as np
import argparse
import warnings
from tqdm import tqdm

from model import AnoModel
from sklearn.neighbors import LocalOutlierFactor
from utils import *
from simulate_data import generate_ny_flow

import torch

torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='experiment settings.')

parser.add_argument('--epochs', type=int, default=100, help='training epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    help='Training batch size')
parser.add_argument('--weight', type=int, default=5, help='anomaly weight')
parser.add_argument('--gen_data',
                    type=bool,
                    default=False,
                    help='if generate data')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, flow, label):
        super(Dataset, self).__init__()
        self.feature = feature
        self.flow = flow
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.flow[index], self.label[index]

    def __len__(self):
        return len(self.feature)


def load_data(train_size=0.95):
    data = sio.loadmat("./data/fake_data.mat")
    feature = data["x"]
    flow = data["y"]
    label = data["label"]

    feature = torch.tensor(feature.reshape(-1, feature.shape[-1])).float()
    flow = torch.tensor(flow.reshape(-1, flow.shape[-1])).float()
    label = torch.tensor(label).flatten().float()

    if train_size < 1:
        train_size = int(len(feature) * train_size)

    train_dataset = Dataset(
        feature[:train_size],
        flow[:train_size],
        label[:train_size],
    )
    test_dataset = Dataset(
        feature[train_size:],
        flow[train_size:],
        label[train_size:],
    )
    return train_dataset, test_dataset


def ano_detect(flow, err, stfeature, label):
    # FAL
    points = np.concatenate([stfeature, err], axis=1)
    detector = LocalOutlierFactor(n_neighbors=100, novelty=True)
    detector.fit(points)
    ano_scores = -detector.decision_function(points)
    compute_metrics(ano_scores, label, "FAL")

    # LOF
    points = flow
    detector = LocalOutlierFactor(n_neighbors=100, novelty=True)
    detector.fit(points)
    ano_scores = -detector.decision_function(points)
    compute_metrics(ano_scores, label, "LOF")


if __name__ == "__main__":
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if args.gen_data:
        generate_ny_flow(args.weight)

    model = AnoModel(
        tf_units=[128, 16],
        sf_units=[],
        st_units=[128, 256, 64],
        out_dim=4,
        tf_dim=36,
        sf_dim=16,
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataset, test_dataset = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_feature = test_dataset.feature.to(device)
    test_flow = test_dataset.flow.to(device)

    for epoch in range(EPOCHS):
        index_list = []
        model.train()
        for feature, flow, _ in tqdm(
                train_loader,
                desc="Epoch {}".format(epoch + 1),
        ):
            feature = feature.to(device)
            flow = flow.to(device)

            shuffled_index = np.arange(feature.shape[0])
            np.random.shuffle(shuffled_index)
            index_list.append(shuffled_index)
            loss = model.construct_loss(
                feature,
                feature[shuffled_index],
                flow,
                flow[shuffled_index],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        sum_loss = 0.
        for i, (feature, flow, _) in enumerate(train_loader):
            feature = feature.to(device)
            flow = flow.to(device)

            loss = model.construct_loss(
                feature,
                feature[index_list[i]],
                flow,
                flow[index_list[i]],
            )
            sum_loss += loss.item()
        print("Training loss : {}".format(sum_loss / len(train_loader)))

    model.eval()
    err, stfeature = model.decompose(test_feature, test_flow)

    ano_detect(
        test_flow.detach().cpu().numpy(),
        err.detach().cpu().numpy(),
        stfeature.detach().cpu().numpy(),
        test_dataset.label.numpy(),
    )
