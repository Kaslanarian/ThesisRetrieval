import torch
from torch import nn
import torch.nn.functional as F
import scipy.stats as stats


class AnoModel(nn.Module):
    def __init__(
        self,
        tf_units: list,
        sf_units: list,
        st_units: list,
        out_dim: int,
        sf_dim: int,
        tf_dim: int,
        _lambda: float = 0.001,
    ) -> None:
        super().__init__()
        self._tf_units = tf_units
        self._sf_units = sf_units
        self._st_units = st_units

        self._out_dim = out_dim
        self._sf_dim = sf_dim
        self._tf_dim = tf_dim

        self._lambda = _lambda

        self.t_net = nn.Sequential()
        size_list = [self._tf_dim] + self._tf_units
        for i, size in enumerate(size_list[:-1]):
            layer = nn.Linear(size, size_list[i + 1])
            self.t_net.add_module("tf_dense_{}".format(i), layer)
            self.t_net.add_module("tf_act_{}".format(i), nn.ReLU())

        self.s_net = nn.Sequential()
        size_list = [self._sf_dim] + self._sf_units
        for i, size in enumerate(size_list[:-1]):
            self.s_net.add_module(
                "sf_dense_{}".format(i),
                nn.Linear(size, size_list[i + 1]),
            )
            self.s_net.add_module("sf_act_{}".format(i), nn.ReLU())

        self.st_net = nn.Sequential()
        size_list = [
            (self._tf_units[-1] if len(self._tf_units) > 0 else self._tf_dim) +
            (self._sf_units[-1] if len(self._sf_units) > 0 else self._sf_dim)
        ] + self._tf_units
        for i, size in enumerate(size_list[:-1]):
            self.st_net.add_module(
                "sf_dense_{}".format(i),
                nn.Linear(size, size_list[i + 1]),
            )
            self.st_net.add_module("sf_act_{}".format(i), nn.ReLU())

        affine = nn.Linear(size_list[-1], self._out_dim)
        trunc_normal = stats.truncnorm(
            -0.2,
            0.2,
            loc=0,
            scale=0.1,
        )
        affine.weight = nn.parameter.Parameter(
            torch.tensor(trunc_normal.rvs(affine.weight.shape)).float())
        affine.bias = nn.parameter.Parameter(
            torch.tensor(trunc_normal.rvs(affine.bias.shape)).float())
        self.st_net.add_module("sf_dense_{}".format(i + 1), affine)

    def forward(self, tfeature, sfeature):
        t_output = self.t_net(tfeature)
        s_output = self.s_net(sfeature)
        st_feature = torch.concat([t_output, s_output], axis=1)
        st_output = self.st_net(st_feature)
        return st_output, st_feature

    def construct_loss(self, x0, x1, y0, y1):
        tfeature0 = x0[:, :self._tf_dim]
        sfeature0 = x0[:, -self._sf_dim:]
        tfeature1 = x1[:, :self._tf_dim]
        sfeature1 = x1[:, -self._sf_dim:]

        y_pred0, stfeature0 = self.forward(tfeature0, sfeature0)
        y_pred1, stfeature1 = self.forward(tfeature1, sfeature1)

        pred_loss = F.mse_loss(y_pred0, y0) + F.mse_loss(y_pred1, y1)
        output_dis = F.mse_loss(y_pred0, y_pred1)
        input_dis = F.mse_loss(stfeature0, stfeature1)

        simi_loss = self._lambda * output_dis / (input_dis + 1)
        return pred_loss + simi_loss

    def decompose(self, feature, flow):
        tfeature = feature[:, :self._tf_dim]
        sfeature = feature[:, -self._sf_dim:]
        y_pred, stfeature = self.forward(tfeature, sfeature)
        return y_pred - flow, stfeature