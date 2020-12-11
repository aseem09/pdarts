import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Architect(object):

    def __init__(self, model_1, model_2, network_params_1, network_params_2, criterion, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model_1 = model_1
        self.model_2 = model_2
        self.network_params_1 = network_params_1
        self.network_params_2 = network_params_2
        self.criterion = criterion
        self.args = args

    def step(self, input_train, target_train, input_valid, target_valid, lr, optimizer_a_1, optimizer_a_2):
        optimizer_a_1.zero_grad()
        optimizer_a_2.zero_grad()

        self._backward_step_unrolled(
            input_train, target_train, input_valid, target_valid, lr, None)

        nn.utils.clip_grad_norm_(self.model_1.module.arch_parameters(), self.args.grad_clip)
        nn.utils.clip_grad_norm_(self.model_2.module.arch_parameters(), self.args.grad_clip)
        optimizer_a_1.step()
        optimizer_a_2.step()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, network_optimizer):
        # unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        # unrolled_loss.backward()

        logits_1 = self.model_1(input_valid)
        logits_2 = self.model_2(input_valid)

        loss_1 = self.criterion(logits_1, target_valid)
        loss_2 = self.criterion(logits_2, target_valid)

        loss_w = loss_1 + loss_2
        loss_w.backward()
        
        vector_1 = [v.grad.data for v in self.network_params_1]
        vector_2 = [v.grad.data for v in self.network_params_2]
        
        implicit_grads_a1 = self._hessian_vector_product_model_1(vector_1, vector_2, input_train, target_train)
        implicit_grads_a2 = self._hessian_vector_product_model_2(vector_1, vector_2, input_train, target_train)

        for i, v in enumerate(self.model_1.module.arch_parameters()):
            data = implicit_grads_a1[i] * (-1) * (lr)
            if v.grad is None:
                v.grad = Variable(data)
            else:
                v.grad.data.copy_(data)

        for i, v in enumerate(self.model_2.module.arch_parameters()):
            data = implicit_grads_a2[i] * (-1) * (lr)
            if v.grad is None:
                v.grad = Variable(data)
            else:
                v.grad.data.copy_(data)

        # for i, param in enumerate(self.model_1.module.arch_parameters()):
        #         print(param.grad)

    # For model 1
    def _hessian_vector_product_model_1(self, vector_1, vector_2, input, target, r=1e-2):
        R = r / _concat(vector_1).norm()
        for p, v in zip(self.network_params_1, vector_1):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model_1.module.arch_parameters())

        for p, v in zip(self.network_params_1, vector_1):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model_1.module.arch_parameters())

        for p, v in zip(self.network_params_1, vector_1):
            p.data.add_(R, v)

        ig_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        R = r / _concat(vector_2).norm()
        for p, v in zip(self.network_params_2, vector_2):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model_1.module.arch_parameters())

        for p, v in zip(self.network_params_2, vector_2):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model_1.module.arch_parameters())

        for p, v in zip(self.network_params_2, vector_2):
            p.data.add_(R, v)

        ig_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        return [(x-y) for x, y in zip(ig_1, ig_2)]

    # For model 2
    def _hessian_vector_product_model_2(self, vector_1, vector_2, input, target, r=1e-2):
        R = r / _concat(vector_1).norm()
        for p, v in zip(self.network_params_1, vector_1):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model_2.module.arch_parameters())

        for p, v in zip(self.network_params_1, vector_1):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model_2.module.arch_parameters())

        for p, v in zip(self.network_params_1, vector_1):
            p.data.add_(R, v)

        ig_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        R = r / _concat(vector_2).norm()
        for p, v in zip(self.network_params_2, vector_2):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model_2.module.arch_parameters())

        for p, v in zip(self.network_params_2, vector_2):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model_2.module.arch_parameters())

        for p, v in zip(self.network_params_2, vector_2):
            p.data.add_(R, v)

        ig_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        return [(x-y) for x, y in zip(ig_1, ig_2)]

    def compute_loss(self, input, target):
        logits_1 = self.model_1(input)
        loss_1 = self.criterion(logits_1, target)

        logits_2 = self.model_2(input)
        loss_2 = self.criterion(logits_2, target)

        logits_1 = F.softmax(logits_1, dim=1)
        logits_2 = F.softmax(logits_2, dim=1)
        logits_3 = torch.log10(F.softmax(logits_1, dim=1))
        logits_4 = torch.log10(F.softmax(logits_2, dim=1))

        loss_add = torch.sum(logits_2*logits_3*-1) + torch.sum(logits_1*logits_4*-1)

        loss = (loss_1 + loss_2) + self.args.c_lambda*(loss_add)
        return loss