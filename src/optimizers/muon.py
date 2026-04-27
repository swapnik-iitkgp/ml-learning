import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                m = state['momentum']

                # Weight decay
                if wd != 0:
                    grad = grad + wd * p

                # Momentum update
                m.mul_(beta).add_(grad, alpha=1 - beta)

                # Muon-style normalization
                update = m / (m.norm() + eps)

                # Parameter update
                p.add_(update, alpha=-lr)

        return loss
