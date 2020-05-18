import torch
import torch.nn as nn
from sparsemax import Sparsemax

class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=15, alpha = 1., epsilon=1.0e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1) # Sparsemax(dim=-1)  # nn.Softmax(dim=1)

    def forward(self, s, is_norm=True, is_log=False):
        if is_log:
            return self.forward_log(s, is_norm)
        else: 
            return self.forward_pai(s, is_norm)

    def forward_origin(self, s, is_norm): 
        s = self.softmax(self.alpha*s) ## voting layer
        if is_norm:  
            s = s + self.epsilon
            for i in range(self.max_iter):
                if i % 2 == 1:
                    # column norm
                    sum_s = torch.sum(s, dim=2, keepdim=True)
                else:
                    # row norm
                    sum_s = torch.sum(s, dim=1, keepdim=True)
                s = s.div(sum_s)
        return s

    def forward_pai(self, s, is_norm): 
        for i in range(self.max_iter):
            if i % 2 == 1:
                # column norm
                sum_s = torch.sum(s, dim=2, keepdim=True)
            else:
                # row norm
                sum_s = torch.sum(s, dim=1, keepdim=True)
            sum_s = sum_s.expand_as(s)
            s = s.div(sum_s.expand_as(s) + torch.full_like(sum_s, self.epsilon))
        return s

    def forward_log(self, s, is_norm):
        s = self.alpha*s ## voting layer
        s = s - torch.logsumexp(s, 1, keepdim=True)
        if is_norm:
            # operations are performed on log_s
            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(s, 2, keepdim=True)
                else:
                    log_sum = torch.logsumexp(s, 1, keepdim=True)
                s = s - log_sum
        return torch.exp(s)


class GumbelSinkhorn(nn.Module):
    """
    GumbelSinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=9, alpha = 1, epsilon=1.0e-10):
        super(GumbelSinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.sinkhorn = Sinkhorn(max_iter=max_iter, alpha=alpha, epsilon=epsilon)

    def sample_gumbel(self, t_like, eps=1e-20):
        """
        randomly sample standard gumbel variables
        """
        u = torch.empty_like(t_like).uniform_()
        return -torch.log(-torch.log(u + eps) + eps) 

    def forward(self, s, is_norm=True, is_log=True):
        s = s + self.sample_gumbel(s)  
        s = self.sinkhorn(s, is_norm=is_norm, is_log=is_log)
        return s
