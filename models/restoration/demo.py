import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from optimizer import get_optimizer
from scheduler import get_scheduler
from mscv import ExponentialMovingAverage, AverageMeters
from utils.res_metrics import tensor2im,quality_assess

class BLOCK(nn.Module):
    def __init__(self):
        super(BLOCK,self).__init__()
        self.conv1 = nn.Conv2d(64,64,3,padding=1)
        self.ac = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        fea = self.conv1(x)
        fea = self.ac(fea)
        fea = self.conv2(fea)
        return fea + x
class demo(nn.Module):
    def __init__(self):
        super(demo,self).__init__()
        self.preconv = nn.Conv2d(3,64,3,padding=1)
        self.ac = nn.ReLU()
        self.block = BLOCK()
        self.finalconv = nn.Conv2d(64,3,3,padding=1)
    def forward(self,x):
        fea = self.preconv(x)
        for i in range(10):
            fea = self.block(fea)
        return self.finalconv(fea)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.net = demo()
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

if __name__=="__main__":
    model = conv_block_nested(3,256,3)
    input = torch.randn([4,3,256,256])
    output = model(input)
    print(output.shape)
