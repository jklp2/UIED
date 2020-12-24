from torch.utils.data import DataLoader
from data.datasets import uwdataset, collate_fn
from utils.visual import get_summary_writer, visualize_boxes
from tqdm import tqdm
from mscv import write_meters_loss, write_image
from models.det.faster_rcnn import Model
import numpy as np
import ipdb

classes = ['0B', '1B', '2B']
dataset = uwdataset("/media/raid/underwater/chinamm2019uw/chinamm2019uw_train",
                    "/media/windows/c/datasets/underwater/UIEBD")
dl = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=True, collate_fn=collate_fn)
dataset_val = uwdataset("/media/raid/underwater/chinamm2019uw/chinamm2019uw_train",
                        "/media/windows/c/datasets/underwater/UIEBD", istrain=False)
dl_val = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle=True, collate_fn=collate_fn)

writer = get_summary_writer('logs','preview')
idx = 0
pbar = tqdm(dl)
for data in pbar:
    pbar.set_description("preview")
    det_image = data['det_img'][0].detach().cpu().numpy().transpose([1,2,0])
    det_image = (det_image.copy()*255).astype(np.uint8)

    uieb_inp = data['uieb_inp'][0].detach().cpu().numpy().transpose([1,2,0])
    uieb_inp = (uieb_inp.copy()*255).astype(np.uint8)

    uieb_ref = data['uieb_inp'][0].detach().cpu().numpy().transpose([1, 2, 0])
    uieb_ref = (uieb_ref.copy() * 255).astype(np.uint8)

    bboxes = data['det_bboxes'][0].cpu().numpy()
    labels = data['det_labels'][0].cpu().numpy().astype(np.int32)
    visualize_boxes(image=det_image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(100, 101, size=[len(bboxes)])/100), class_labels=classes)
    write_image(writer, f'preview_mm2019/{idx}', 'image', det_image, 0, 'HWC')
    write_image(writer,f'preview_uieb/{idx*2}','image',uieb_inp,0,'HWC')
    write_image(writer,f'preview_uieb/{idx*2+1}','image',uieb_ref,0,'HWC')

    idx+=1

    # print(data["uieb_inp"].shape)

writer.flush()

det_model = Model().cuda()
writer = get_summary_writer('logs', 'debug')
steps = 0
for epoch in range(200):
    dl.dataset.reset()
    it = 0
    pbar = tqdm(dl)
    for data in pbar:
        if it == 0:
            print(data["det_path"])
        pbar.set_description('epoch: %i' % epoch)
        it += 1
        det_model.update(data)
        if it % 10 == 0:
            write_meters_loss(writer, 'train', det_model.avg_meters, steps)
        steps += 1
        pbar.set_postfix(**(det_model.avg_meters.dic))
    det_model.eval()
    det_model.evaluate(dl_val, epoch, writer, None)
    det_model.train()
ipdb.set_trace()