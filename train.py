from torch.utils.data import DataLoader
from data.datasets import uwdataset, collate_fn, mmvaldataset, uiebvaldataset
from utils.visual import get_summary_writer, visualize_boxes
from tqdm import tqdm
from mscv import write_meters_loss, write_image
from models.det.faster_rcnn import Model as det_Model
from models.restoration.demo import Model as res_Model
import numpy as np
from utils.res_metrics import tensor2im
from os.path import join
import ipdb

classes = ['0B', '1B', '2B']
root = "/media/windows/c/datasets/underwater"
dataset = uwdataset(join(root,"chinamm2019uw/chinamm2019uw_train"),
                    join(root,"UIEBD"))
dl = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True, collate_fn=collate_fn)

dataset_val = mmvaldataset(join(root,"chinamm2019uw/chinamm2019uw_train"))
dl_val = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle=True, collate_fn=collate_fn)

dataset_val_uieb = uiebvaldataset(join(root,"UIEBD"))
dl_val_uieb = DataLoader(dataset_val_uieb, batch_size=1, num_workers=4, shuffle=True, collate_fn=collate_fn)


# writer = get_summary_writer('logs','preview')
# idx = 0
# pbar = tqdm(dl)
# for data in pbar:
#     pbar.set_description("preview")
#     det_image = data['det_img'][0].detach().cpu().numpy().transpose([1,2,0])
#     det_image = (det_image.copy()*255).astype(np.uint8)
#
#     uieb_inp = data['uieb_inp'][0].detach().cpu().numpy().transpose([1,2,0])
#     uieb_inp = (uieb_inp.copy()*255).astype(np.uint8)
#
#     uieb_ref = data['uieb_ref'][0].detach().cpu().numpy().transpose([1, 2, 0])
#     uieb_ref = (uieb_ref.copy() * 255).astype(np.uint8)
#
#     bboxes = data['det_bboxes'][0].cpu().numpy()
#     labels = data['det_labels'][0].cpu().numpy().astype(np.int32)
#     visualize_boxes(image=det_image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(100, 101, size=[len(bboxes)])/100), class_labels=classes)
#     write_image(writer, f'preview_mm2019/{idx}', 'image', det_image, 0, 'HWC')
#     write_image(writer,f'preview_uieb/{idx*2}','image',uieb_inp,0,'HWC')
#     write_image(writer,f'preview_uieb/{idx*2+1}','image',uieb_ref,0,'HWC')
#
#     idx+=1
#
#     # print(data["uieb_inp"].shape)
#
# writer.flush()

det_model = det_Model().cuda()
res_model = res_Model().cuda()
writer = get_summary_writer('logs', 'debug')
steps = 0
for epoch in range(2000):
    dl.dataset.reset()
    it = 0
    pbar = tqdm(dl)
    for data in pbar:
        if it == 0:
            print(data["det_path"])
        pbar.set_description('epoch: %i' % epoch)
        it += 1
        # det_model.update(data)
        res_model.update(data)
        # if it % 10 == 0:
        #     write_meters_loss(writer, 'train_det', det_model.avg_meters, steps)
        #     write_meters_loss(writer, 'train_res', res_model.avg_meters, steps)
        steps += 1
        # pbar.set_postfix(**{"det_loss":det_model.avg_meters.dic["loss"],"res_loss":res_model.avg_meters.dic["loss"]})
        if it%200==0:
            write_image(writer, f'train/{epoch*1000+it }', 'image', tensor2im(res_model.input), 0, 'HWC')
            write_image(writer, f'train/{epoch*1000+it+ 1}', 'image', tensor2im(res_model.ref), 0, 'HWC')
            write_image(writer, f'train/{epoch*1000+it+ 2}', 'image', tensor2im(res_model.output), 0, 'HWC')

    if epoch % 100 == 99:
        # det_model.eval()
        # det_model.evaluate(dl_val, epoch, writer, None)
        # det_model.train()
        res_model.eval()
        res_model.evaluate(dl_val_uieb)
        res_model.train()
ipdb.set_trace()
