#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from pyepo.model.grb.vrp import VRP



# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item * num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":

    # generate data
    num_data = 100 # number of data
    num_feat = 100 # size of feature
    num_item = 25 # number of items
    x, c = pyepo.data.vrp.genData(num_data, num_feat, num_item, seed=135)

    # init optimization model
    optmodel = VRP(num_item)

    # init prediction model
    predmodel = LinearRegression()
    # set optimizer
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
    # init SPO+ loss
    spop = pyepo.func.SPOPlus(optmodel, processes=4)

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # training
    num_epochs = 100
    for epoch in range(num_epochs):
        for data in dataloader:
            x, c, w, z = data
            # forward pass
            cp = predmodel(x)
            loss = spop(cp, c, w, z).mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # eval
    regret = pyepo.metric.regret(predmodel, optmodel, dataloader)
    print("Regret on Training Set: {:.4f}".format(regret))
