import torch
import torch.nn as nn
from Models.Layers.graph import *
import torch.nn.functional as F


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        # 这里应该做的是时间的卷积
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * kernel_size,  # 这里我还没有理解
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)  # 做时空卷积
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # 这里应该是对空间做卷积，还没有完全弄懂
        return x.contiguous(), A  # x.contiguous()是因为做了einsum后，x空间存储方式并不是"连续的


class st_gcn_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # 当stride=1时，可以维持第1维size保持不变

        self.gcn = ConvTemporalGraphical(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size[1])  # 所以kernel_size[1]是gcn的kernel size?
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1),
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:  # input 和 output的size一致
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),  # 不需要padding的原因：kernel_size = 1，不会改变每一个feature map的size
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x, A = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return x, A


class ST_GCN_18(nn.Module):
    def __int__(self,
                in_channels,
                num_classes,
                graph_cfg,
                edge_importance_weighting=True,
                data_bn=True,
                **kwargs):
        super(ST_GCN_18, self).__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        # 邻接矩阵
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else lambda x: x
        # kwargs0 将 dropout 从 kwargs 中剔除
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList([
            st_gcn_block(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=1, residual=False,
                         **kwargs0),
            st_gcn_block(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=2, **kwargs),
            st_gcn_block(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2, **kwargs),
            st_gcn_block(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1, **kwargs),
            st_gcn_block(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1, **kwargs)
        ])

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            # 给每一层都配一个 M
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size()), requires_grad=True) for _ in range(len(self.st_gcn_networks))])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, A):
        # data normalization
        # M is the number of instance in a frame
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # 我觉得这里是将每个关节的每个channel的时间层和空间层做一个平均
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x
