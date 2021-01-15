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




class unique(nn.Module):
    def __init__(self, n):
        super().__init__()
        chan = 48
        self.ac = nn.ReLU()
        self.st = nn.ModuleList([ResNetLayer(chan, 64, kernel_size=3) for _ in range(n)])
        self.pre_conv = nn.Conv2d(3, chan, kernel_size=3, bias=True, padding=1)
        self.final_conv = nn.Conv2d(chan, 3, kernel_size=3, bias=True, padding=1)

    def forward(self, inp):
        x = self.ac(self.pre_conv(inp))
        z = x.clone().detach().requires_grad_()
        for f in self.st:
            z = f(z, x)
        return self.ac(self.final_conv(z))

class repeat(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        chan = 48
        self.ac = nn.ReLU()
        self.f = ResNetLayer(chan, 64, kernel_size=3)
        self.pre_conv = nn.Conv2d(3, chan, kernel_size=3, bias=True, padding=1)
        self.final_conv = nn.Conv2d(chan, 3, kernel_size=3, bias=True, padding=1)

    def forward(self, inp):
        x = self.ac(self.pre_conv(inp))
        z = x.clone().detach().requires_grad_()
        for i in range(self.n):
            z = f(z, x)
        return self.ac(self.final_conv(z))




class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        chan = 48
        f = ResNetLayer(chan, 64, kernel_size=3)
        self.net = unique(5).to(device)
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
