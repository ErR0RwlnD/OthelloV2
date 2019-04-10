import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyper import Hyper
import numpy as np


class _Resnet(nn.Module):
    def __init__(self, game, dropout=0.3):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.dropout = dropout

        super(_Resnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, 512 * (self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))),
                      p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))),
                      p=self.dropout, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class GeneratorNet():
    def __init__(self, game):
        self.net = _Resnet(game, 0.3)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if Hyper.cuda:
            self.net.cuda()

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if Hyper.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_checkpoint(self, filename, folder=Hyper.checkpoints, strict=True):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise('No such checkpoint')
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint['state_dict'], strict=strict)
