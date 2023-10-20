import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc as voc
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg import PTVIT, VIT_MOD
from model.backbone import vit_base_patch16_224
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, \
    crop_from_roi_neg
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--pretrained_pth",
                    default='/data/c425/tjf/PTVIT/pretrained/checkpoints/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                    type=str,
                    help="pooling choice for patch tokens")

parser.add_argument("--data_folder", default='/data/c425/tjf/datasets/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=20, type=int, help="number of classes")
parser.add_argument("--crop_size", default=224, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

# dir
parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str, help="work_dir_voc_wseg")
parser.add_argument("--tensorboard_dir", default="tensorboard_dir", type=str, help="tensorboard_dir")
parser.add_argument("--visual_dir", default="visual_dir", type=str, help="visualization attention map dir")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=2, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='CosWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="power factor for poly scheduler")

parser.add_argument("--max_iters", default=100000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--temp", default=0.5, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")
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


def visual_attmap(model=None, data_loader=None, device=None, args=None):
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            img_name, inputs, cls_label, img_box, crops = data
            inputs = inputs.to(device, non_blocking=True)
            cls_label = cls_label.to(device, non_blocking=True)

            attn_weights = model(inputs, visual_attmap_f=True)
            attn_weights = torch.stack(attn_weights)  # BLK * BAT * Head * tokens * tokens
            print(f'attn_weights: {attn_weights.shape}')


def validate(model=None, data_loader=None, args=None):
    model.eval()
    avg_meter = AverageMeter()
    mAP = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls = model(inputs)
            # print(f'val_cls: {cls}')    # 1*20
            cls = torch.sigmoid(cls)

            # cls_pred = (cls > 0).type(torch.int16)
            # print(f'val_cls_pred: {cls_pred}')  # 1*20
            # print(f'cls_label: {cls_label}')    # 1*20
            mAP_b = evaluate.compute_mAP(cls_label.cpu().numpy(), cls.cpu().numpy())
            # mAP = mAP + mAP_b
            # _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({'cls_mAP': mAP_b[0]})
    cls_mAP = avg_meter.pop('cls_mAP')
    model.train()

    return cls_mAP


def train(args=None):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    # pt_model = PTVIT(num_classes=args.num_classes)
    # model = pt_model
    # new_model = vit_base_patch16_224(num_classes=20)
    # model = new_model

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

        # # interpolate position embedding
        # pos_embed_checkpoint = checkpoint_model['pos_embed']    # 1*197*768
        # embedding_size = pos_embed_checkpoint.shape[-1]  # 768
        # num_patches = model.patch_embed.num_patches  # patches 块数
        # if args.pretrained_pth.startswith('https'):
        #     num_extra_tokens = 1    # 载入官方给出的权重，额外tokens指代原来的cls_token
        # else:
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches   # 否则指代多类token
        #
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # 14
        #
        # new_size = int(num_patches ** 0.5)      # 28
        #
        # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]  # 原来的位置编码
        # pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # pos_tokens = torch.nn.functional.interpolate(
        #     pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)  # 双线性插值
        # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # checkpoint_model['pos_embed'] = pos_tokens  # 再放回参数pth中

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    pt_optim = getattr(optimizer, args.optimizer)(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power
    )

    if args.local_rank == 0:
        writer = SummaryWriter(args.tensorboard_dir)

    logging.info('\nOptimizer: \n%s' % pt_optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    model.train()

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)

    if args.visual_attmap:
        visual_attmap(model, train_loader, device, args)
        return

    avg_meter = AverageMeter()
    val_cls_mAP_best = 0.0

    for n_iter in range(args.max_iters):
        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

        pred = model(inputs)

        loss = F.multilabel_soft_margin_loss(pred, cls_label)

        avg_meter.add({
            'cls_loss': loss.item(),
        })

        pt_optim.zero_grad()
        loss.backward()
        pt_optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = pt_optim.param_groups[0]['lr']

            if args.local_rank == 0:
                loss_log = avg_meter.pop('cls_loss')
                logging.info(
                    "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.15f" % (
                        n_iter + 1, delta, eta, cur_lr, loss_log,
                        ))
                writer.add_scalars('train/lr', {"lr:": cur_lr}, global_step=n_iter)
                writer.add_scalars('train/cls_loss', {"cls_loss:": loss_log}, global_step=n_iter)

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)

            val_cls_mAP = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                if val_cls_mAP > val_cls_mAP_best:
                    val_cls_mAP_best = val_cls_mAP
                    best_ckpt_name = os.path.join(args.ckpt_dir, "1_val_cls_mAP_best_model_iter_%d.pth" % (n_iter + 1))
                    torch.save(model.state_dict(), best_ckpt_name)
                logging.info("val cls mAP: %.6f" % val_cls_mAP)
                writer.add_scalars('val/cls mAP', {"cls mAP:": val_cls_mAP}, global_step=n_iter)

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tensorboard_dir = os.path.join(args.tensorboard_dir, timestamp)
    args.visual_dir = os.path.join(args.visual_dir, timestamp)

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    # fix random seed
    setup_seed(args.seed)
    train(args=args)
