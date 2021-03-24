import torch
import torch.nn.functional as F


# non-differentiable topk functions for comparison

def topk_items(x, k):
    # [b, np, sq, sk] --> [b, np, sq*sk]
    orig_shape = x.size()
    x = x.view(x.size(0), x.size(1), x.size(-1) * x.size(-2))
    values, _ = torch.topk(x, k)  # get top k items in each row
    top = values[:, :, -1].unsqueeze(2).expand_as(x)  # broadcast smallest topk item in row across the whole row
    x = x.masked_fill(torch.lt(x, top), float('-inf')).type_as(x)  # masks all values < smallest topk item to -inf
    x = x.view(*orig_shape)
    return x


def topk_items_rowwise(x, k):
    # [b, np, sq, sk]
    values, _ = x.topk(k, dim=-1) # get top k items in each row
    top = values[..., -1].unsqueeze(-1).expand_as(x)  # broadcast smallest topk item in row across the whole row
    mask = x < top
    x = x.masked_fill(mask, -torch.finfo(x.dtype).max)  # masks all values < smallest topk item to -inf
    return x


# support functions for differentiable topk attention, adapted from https://arxiv.org/pdf/2002.06504.pdf

def sinkhorn_forward(x, mu, nu, eps, max_iter):
    assert max_iter > 0
    bs, n, k_ = x.size()
    v = torch.ones([bs, 1, k_]) / (k_)
    γ = torch.exp(-x / eps)
    if torch.cuda.is_available():
        v = v.cuda()
    for i in range(max_iter):
        u = mu / (γ * v).sum(-1, keepdim=True)
        v = nu / (γ * u).sum(-2, keepdim=True)
    gamma = u * γ * v
    return gamma


def sinkhorn_forward_stabilized(x, mu, nu, eps, max_iter):
    assert max_iter > 0
    bs, n, k_ = x.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, k + 1])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()

    epsilon_log_mu = eps * torch.log(mu)
    epsilon_log_nu = eps * torch.log(nu)

    def min_epsilon_row(z, eps):
        return -eps * torch.logsumexp((-z) / eps, -1, keepdim=True)

    def min_epsilon_col(z, eps):
        return -eps * torch.logsumexp((-z) / eps, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(x - g, eps) + epsilon_log_mu
        g = min_epsilon_col(x - f, eps) + epsilon_log_nu

    γ = torch.exp((-x + f + g) / eps)
    return γ


def sinkhorn_backward(grad_output_gamma, gamma, mu, nu, eps):
    nu_ = nu[:, :, :-1]
    gamma_ = gamma[:, :, :-1]

    bs, n, k_ = gamma.size()

    inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
    kappa = torch.diag_embed(nu_.squeeze(-2)) - \
            torch.matmul(gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), gamma_)  # [bs, k, k]

    inv_kappa = torch.inverse(kappa)  # [bs, k, k]
    gamma_mu = inv_mu.unsqueeze(-1) * gamma_
    l = gamma_mu.matmul(inv_kappa)  # [bs, n, k]
    G1 = grad_output_gamma * gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * gamma  # [bs, n, k+1]
    g1_l = g1.unsqueeze(-2).matmul(l)  # [bs, 1, k]
    G22 = g1_l.matmul(gamma_mu.transpose(-1, -2)).transpose(-1, -2) * gamma  # [bs, n, k+1]
    G23 = - F.pad(g1_l, pad=(0, 1), mode='constant', value=0) * gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = - l.matmul(g2) * gamma  # [bs, n, k+1]
    G32 = F.pad(inv_kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode='constant', value=0) * gamma  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, b, k+1]

    grad_x = (-G1 + G2 + G3) / eps  # [bs, n, k+1]
    return grad_x


class DifferentiableTopKFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mu, nu, eps, max_iter):
        with torch.no_grad():
            if eps > 1e-2:
                gamma = sinkhorn_forward(x, mu, nu, eps, max_iter)
                if bool(torch.any(gamma != gamma)):
                    print('NaN appeared in gamma - recomputing...')
                    gamma = sinkhorn_forward_stabilized(x.mu.nu, eps, max_iter)
            else:
                gamma = sinkhorn_forward_stabilized(x, mu, nu, eps, max_iter)
            ctx.save_for_backward(mu, nu, gamma)
            ctx.eps = eps
        return gamma

    @staticmethod
    def backward(ctx, grad_output_gamma):
        eps = ctx.eps
        mu, nu, gamma = ctx.saved_tensors
        with torch.no_grad():
            grad_x = sinkhorn_backward(grad_output_gamma, gamma, mu, nu, eps)
        return grad_x, None, None, None, None


class DifferentiableTopK(torch.nn.Module):

    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(DifferentiableTopK, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        print('anchors: ', self.anchors)
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)
        x = (scores - self.anchors) ** 2
        x = x / (x.max().detach())

        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()

        gamma = DifferentiableTopKFunc.apply(x, mu, nu, self.epsilon, self.max_iter)
        A = gamma[:, :, :self.k] * n

        return A, None
