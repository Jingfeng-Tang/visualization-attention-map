import argparse
import datetime
import logging
import math
import os
import random
import sys

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc as voc
from model.model_seg_neg import PTVIT, VIT_MOD
from model.backbone import vit_base_patch16_224
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import Image

torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="model backbone")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--pretrained_pth",
                    default='/data/c425/tjf/PTVIT/work_dir_voc/2023-10-17-22-09-58-860791/checkpoints/1_val_cls_mAP_best_model_iter_90000.pth',
                    type=str,
                    help="pooling choice for patch tokens")

parser.add_argument("--data_folder", default='/data/c425/tjf/datasets/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=20, type=int, help="number of classes")
parser.add_argument("--crop_size", default=224, type=int, help="crop_size in training")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

# dir
parser.add_argument("--tensorboard_dir", default="tensorboard_dir", type=str, help="tensorboard_dir")
parser.add_argument("--visual_dir", default="visual_dir", type=str, help="visualization attention map dir")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=1, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--visual_attmap", action="store_true", help="visual_attmap")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def show_cam_on_image(img, mask, save_path=None):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap.transpose((2, 0, 1))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def visual_attmap(model=None, data_loader=None, device=None, args=None):
    model.eval()
    if args.local_rank == 0:
        writer = SummaryWriter(args.visual_tensorboard_dir)
    with torch.no_grad():
        for n_iter, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            img_name, inputs, cls_label, img_box, crops, crop_ori_image, ori_image_t, ori_image_t_n = data
            # inputs = inputs.to(device, non_blocking=True)  # crop
            inputs = ori_image_t_n.to(device, non_blocking=True)

            token_w = inputs.shape[2] // 16
            token_h = inputs.shape[3] // 16

            attn_weights = model(inputs, visual_attmap_f=True)
            attn_weights = torch.stack(attn_weights)  # BLK * BAT * Head * tokens * tokens  # 12*1*12*197*197
            attn_weights = attn_weights.squeeze(1)  # 12*12*197*197
            attn_weights = torch.mean(attn_weights, dim=1)  # 12*197*197
            # att_map = attn_weights[11, 0, 1:]  # 先取最后一个block输出的所有头的平均权重
            att_map = attn_weights[-3:].sum(0)[0, 1:]  # 先取最后三个block输出的所有头的平均权重 wh
            # print(att_map.shape)
            # a=[]
            # b=a[10]

            # att_map = att_map.reshape(int(math.sqrt(att_map.shape[0])), int(math.sqrt(att_map.shape[0])))  # 14*14
            att_map = att_map.reshape(1, token_w, token_h)  # 1*14*14
            # 归一化
            # att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            # att_map = att_map.cpu().numpy()
            # att_ = np.clip(att_map, 0, 1) * 255
            # att_map = np.transpose(cv2.applyColorMap(att_.astype(np.uint8), cv2.COLORMAP_JET), (2, 0, 1))
            att_map_t = att_map.unsqueeze(1)
            cls_attentions = F.interpolate(att_map_t, size=(inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)    # 1*1*224*224
            cls_attention = cls_attentions[0]
            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)

            # crop
            # ori_image_0 = crop_ori_image[0]
            # ori_image_0 = torch.tensor(ori_image_0).permute(2, 0, 1).contiguous()   # 3*224*224
            # att_map = show_cam_on_image(ori_image_0.cpu().numpy(), cls_attention.squeeze(0).cpu().numpy())

            # ori
            ori_image = ori_image_t.squeeze(0).permute(2, 0, 1).contiguous()
            att_map = show_cam_on_image(ori_image.cpu().numpy(), cls_attention.squeeze(0).cpu().numpy())

            if args.local_rank == 0:
                writer.add_image("attention map_"+str(n_iter), att_map, global_step=n_iter)
                att_map = att_map.transpose((1, 2, 0))
                att_map_im = Image.fromarray(att_map)
                att_map_im_name = img_name[0] + '_att_map' + '.jpg'
                att_map_im_pth = os.path.join(args.visual_att_map_dir, att_map_im_name)
                att_map_im.save(att_map_im_pth)


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.visual_dir = os.path.join(args.visual_dir, timestamp)
    args.visual_att_map_dir = os.path.join(args.visual_dir, 'att_map_dir')
    args.visual_tensorboard_dir = os.path.join(args.visual_dir, 'tensorboard_dir')


    if args.local_rank == 0:
        os.makedirs(args.visual_dir, exist_ok=True)
        os.makedirs(args.visual_att_map_dir, exist_ok=True)
        setup_logger(filename=os.path.join(args.visual_dir, 'visual_attmap.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    # fix random seed
    setup_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12_Visual_Dataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    device = torch.device(args.local_rank)

    new_model = VIT_MOD(num_classes=20)
    model = new_model

    if args.pretrained:
        if args.pretrained_pth.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_pth, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained_pth, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    visual_attmap(model, train_loader, device, args)


    # while True:
    #     pass
