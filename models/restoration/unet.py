import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from optimizer import get_optimizer
from scheduler import get_scheduler
from mscv import ExponentialMovingAverage, AverageMeters
from utils.res_metrics import tensor2im,quality_assess
class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_channel=3, out_channel=3):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_channel, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_channel, kernel_size=1)

        self.out = nn.Tanh()

    def forward(self, x):
        shapes=[(x.shape[-2],x.shape[-1])]
        for i in range(3):
            shapes.append((shapes[-1][0]//2,shapes[-1][1]//2))

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0,size=shapes[0],mode='bilinear')], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0,size=shapes[1],mode='bilinear')], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1,size=shapes[0],mode='bilinear')], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0,size=shapes[2],mode='bilinear')], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1,size=shapes[1],mode='bilinear')], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2,size=shapes[0],mode='bilinear')], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0,size=shapes[3],mode='bilinear')], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1,size=shapes[2],mode='bilinear')], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2,size=shapes[1],mode='bilinear')], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3,size=shapes[0],mode='bilinear')], 1))

        output = self.out(self.final(x0_4))
        return output


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.net = conv_block_nested(3,256,3)
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
