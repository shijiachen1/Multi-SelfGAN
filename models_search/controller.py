
# -*- coding: utf-8 -*-
# @Date    : 2019-09-29
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_search.building_blocks_search_orign import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE


class Controller(nn.Module):
    def __init__(self, args, total_stage):
        """
        init
        :param args:
        :param cur_stage: varies from 0 to ...
        """
        super(Controller, self).__init__()
        self.hid_size = args.hid_size
        self.total_cells = total_stage
        self.hx = []
        self.cx = []
        self.tokens = []
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        self.lstm = torch.nn.LSTMCell(self.hid_size, self.hid_size)
        # self.tokens = [len(UP_TYPE), len(CONV_TYPE), len(NORM_TYPE), len(CONV_TYPE), len(NORM_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE)**cur_stage]
        for cur_stage in range(self.total_cells):
            if cur_stage > 0:
                self.tokens.append([len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE)**cur_stage])
            else:
                self.tokens.append([len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE)])
            self.encoders.append(nn.Embedding(sum(self.tokens[cur_stage]), self.hid_size))
            self.decoders.append(nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens[cur_stage]]))

    def initHidden(self, batch_size):
        hidden = torch.cuda.FloatTensor(batch_size, self.hid_size).fill_(0)
        return hidden.requires_grad_(False)

    def forward(self, x, hidden, cur_cell, index):
        if index == 0:
            embed = x
        else:
            embed = self.encoders[cur_cell](x)
        hx, cx = self.lstm(embed, hidden)
        logit = self.decoders[cur_cell][index](hx)

        return logit, (hx, cx)

    def sample(self, batch_size, with_hidden=False):
        # x = self.initHidden(batch_size)
        hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        archs = []
        entropies = []
        selected_log_probs = []
        hiddens = []
        for cur_cell in range(self.total_cells):
            x = self.initHidden(batch_size)
            entropy = []
            actions = []
            selected_log_prob = []
            for decode_idx in range(len(self.decoders[cur_cell])):
                logit, hidden = self.forward(x, hidden, cur_cell, decode_idx)
                prob = F.softmax(logit, dim=-1)  # bs * logit_dim
                log_prob = F.log_softmax(logit, dim=-1)
                entropy.append(-(log_prob * prob).sum(1, keepdim=True))  # list[array(bs * 1)]
                action = prob.multinomial(1)  # list[bs * 1]
                actions.append(action)
                op_log_prob = log_prob.gather(1, action.data)  # list[bs * 1]
                selected_log_prob.append(op_log_prob)
                tokens = self.tokens[cur_cell]
                x = action.view(batch_size) + sum(tokens[:decode_idx])
                x = x.requires_grad_(False)

            hiddens.append(hidden[1])
            archs.append(torch.cat(actions, -1))  # batch_size * len(self.decoders)
            selected_log_probs.append(torch.cat(selected_log_prob, -1))  # list(batch_size * len(self.decoders))
            entropies.append(torch.cat(entropy, -1))  # list(bs * 1)

        # hiddens = torch.cat(hiddens, -1)
        archs = torch.cat(archs, -1)
        selected_log_probs = torch.cat(selected_log_probs, -1)
        entropies = torch.cat(entropies, -1)

        if with_hidden:
            return archs, selected_log_probs, entropies, hiddens

        return archs, selected_log_probs, entropies





