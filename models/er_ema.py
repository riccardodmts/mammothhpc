"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform, dataset)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from copy import deepcopy

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def kl_divergence_stable(logits_student, logits_teacher, temperature=1.0):
    """
    Numerically stable KL divergence for knowledge distillation.
    
    Args:
        logits_student (Tensor): Logits from the student network (B x C).
        logits_teacher (Tensor): Logits from the teacher network (B x C).
        temperature (float): Temperature scaling factor.
        
    Returns:
        Tensor: The KL divergence loss.
    """
    # Apply temperature scaling
    scaled_logits_student = logits_student / temperature
    scaled_logits_teacher = logits_teacher / temperature

    # Compute log probabilities in a numerically stable way
    log_p_student = F.log_softmax(scaled_logits_student, dim=-1)
    log_p_teacher = F.log_softmax(scaled_logits_teacher, dim=-1)
    
    # Compute KL divergence using teacher's soft probabilities
    kl_loss = F.kl_div(log_p_student, log_p_teacher.exp(), reduction='batchmean')

    return kl_loss


class ErEMA(ContinualModel):
    """Continual learning via Experience Replay."""
    NAME = 'er_ema'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, default=0.25, help='Penalty weight.')
        parser.add_argument('--softmax_temp', type=float, default=1.0, help='Temperature of the softmax function.')
        parser.add_argument("--ema", type=float, default=0.996, help='Momentum teacher')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErEMA, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.gamma = self.args.ema
        self.count = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        real_batch_size = inputs.shape[0]
        temp = not_aug_inputs

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_no_aug, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device, return_not_aug=True)
            inputs = torch.cat((inputs, buf_inputs))
            temp = torch.cat((not_aug_inputs, buf_no_aug))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)


        logits = self.old_net(temp)

        loss += self.args.alpha  * kl_divergence_stable(outputs, logits, self.args.softmax_temp)


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        
        if self.gamma > 0.0:
            self.update_teacher()

        return loss.item()

    def begin_task(self, dataset):
        if self.count == 0:
            self.old_net = deepcopy(self.net)
            self.old_net.to(self.device)
            for param in self.old_net.parameters():
                param.requires_grad = False
            self.old_net.train()
        self.count += 1
        
    def update_teacher(self):
        # EMA update for the teacher
        with torch.no_grad():
            m = self.gamma  # momentum parameter
            for param_q, param_k in zip(self.net.parameters(), self.old_net.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
