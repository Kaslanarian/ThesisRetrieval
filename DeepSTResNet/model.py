import torch
from torch import nn


class _residual_unit(nn.Module):
    def __init__(self, bn: bool = False) -> None:
        super().__init__()

        self.relu_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
        ) if bn else nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
        )
        self.relu_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
        ) if bn else nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
        )

    def forward(self, input):
        residual = self.relu_conv1(input)
        residual = self.relu_conv2(residual)
        return input + residual


def ResUnits(repetations: int, bn=False):
    return nn.Sequential(*(_residual_unit(bn) for _ in range(repetations)))


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0), ) + self.shape)


class STresnet(nn.Module):
    def __init__(self,
                 c_conf=(3, 2, 32, 32),
                 p_conf=(1, 2, 32, 32),
                 t_conf=(1, 2, 32, 32),
                 external_dim=28,
                 nb_residual_unit=3,
                 bn=False) -> None:
        super().__init__()
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit

        len_seq, nb_flow, _, _ = self.c_conf
        self.c_net = nn.Sequential(
            nn.Conv2d(nb_flow * len_seq, 64, 3, padding='same'),
            ResUnits(repetations=nb_residual_unit, bn=bn),
            nn.ReLU(),
            nn.Conv2d(64, nb_flow, 3, padding='same'),
        )

        len_seq, nb_flow, _, _ = self.p_conf
        self.p_net = nn.Sequential(
            nn.Conv2d(nb_flow * len_seq, 64, 3, padding='same'),
            ResUnits(repetations=nb_residual_unit, bn=bn),
            nn.ReLU(),
            nn.Conv2d(64, nb_flow, 3, padding='same'),
        )

        len_seq, nb_flow, map_height, map_width = self.t_conf
        self.t_net = nn.Sequential(
            nn.Conv2d(nb_flow * len_seq, 64, 3, padding='same'),
            ResUnits(repetations=nb_residual_unit, bn=bn),
            nn.ReLU(),
            nn.Conv2d(64, nb_flow, 3, padding='same'),
        )

        self.Wc = nn.parameter.Parameter(
            data=torch.randn(map_height, map_width),
            requires_grad=True,
        )
        self.Wp = nn.parameter.Parameter(
            data=torch.randn(map_height, map_width),
            requires_grad=True,
        )
        self.Wt = nn.parameter.Parameter(
            data=torch.randn(map_height, map_width),
            requires_grad=True,
        )

        self.external_net = nn.Sequential(
            nn.Linear(self.external_dim, 10),
            nn.ReLU(),
            nn.Linear(10, nb_flow * map_height * map_width),
            nn.ReLU(),
            Reshape(nb_flow, map_height, map_width),
        )

    def forward(self, Xc, Xp, Xt, ext):
        output_c = self.c_net(Xc)
        output_p = self.p_net(Xp)
        output_t = self.t_net(Xt)
        output_ext = self.external_net(ext)
        X_res = output_c * self.Wc + output_p * self.Wp + output_t * self.Wt
        return torch.tanh(X_res + output_ext)
