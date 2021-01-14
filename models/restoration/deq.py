import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from optimizer import get_optimizer
from scheduler import get_scheduler
from mscv import ExponentialMovingAverage, AverageMeters
from utils.res_metrics import tensor2im,quality_assess


class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        # self.norm1 = nn.InstanceNorm2d(n_inner_channels)
        # self.norm2 = nn.InstanceNorm2d(n_channels)
        # self.norm3 = nn.InstanceNorm2d(n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

        # y = F.relu(self.conv1(z))
        # return F.relu(z + x + self.conv2(y))


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res


import torch.autograd as autograd


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

        return fea + x

from torch.autograd import gradcheck
# run a very small network with double precision, iterating to high precision
f = ResNetLayer(2,2, num_groups=2).double()
deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=500).double()
gradcheck(deq, torch.randn(1,2,3,3).double().requires_grad_(), eps=1e-5, atol=1e-3)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        chan = 48
        f = ResNetLayer(chan, 64, kernel_size=3)
        self.net = nn.Sequential(nn.Conv2d(3, chan, kernel_size=3, bias=True, padding=1),
                              # nn.BatchNorm2d(chan),
                              DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
                              # nn.BatchNorm2d(chan),
                              nn.Conv2d(chan, 3, kernel_size=3, bias=True, padding=1),
                              nn.ReLU(),
                              ).to(device)
        self.optimizer = get_optimizer("adam",self.net)
        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = "checkpoints/unetpp"
        self.device = "cuda:0"
        self.lossfn = nn.L1Loss()
    def forward(self, sample):
        self.input = sample["uieb_inp"].to(self.device)
        self.ref = sample["uieb_ref"].to(self.device)
        self.output = self.net(self.input)

    def getloss(self):
        self.loss = self.lossfn(self.ref, self.output)
        self.avg_meters.update({'loss':  self.loss.item()})
    def update(self, sample):
        self.forward(sample)
        self.getloss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def evaluate(self, dataloader):
        pbar = tqdm(dataloader)
        metric_meters = AverageMeters()
        for data in pbar:
            pbar.set_description("res_eval")
            self.forward(data)
            output = tensor2im(self.output)
            target = tensor2im(self.ref)
            res = quality_assess(output, target, tensor2im(self.input))
            metric_meters.update(res)
            pbar.set_postfix(**{ key:metric_meters.dic[key] / metric_meters.total_num[key] for key in metric_meters.dic.keys()})
        return { key:metric_meters.dic[key] / metric_meters.total_num[key] for key in metric_meters.dic.keys()}
if __name__=="__main__":
    model = conv_block_nested(3,256,3)
    input = torch.randn([4,3,256,256])
    output = model(input)
    print(output.shape)
