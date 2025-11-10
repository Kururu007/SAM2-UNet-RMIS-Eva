import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset
import torchmetrics
from torchmetrics import Accuracy, Dice, Precision
from torchmetrics.classification import BinaryJaccardIndex

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.name == "Lr":
            fmtstr = "{name}={val" + self.fmt + "}"
        else:
            fmtstr = "{name}={val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class SampleMeanBinaryJaccard(torchmetrics.Metric):
    """先对 batch 内每个样本独立计算 Binary Jaccard Index，
       然后对这些样本分数求平均。
    """
    higher_is_better: bool = True  # IoU 越大越好

    def __init__(self, **kwargs):
        super().__init__(dist_sync_on_step=False)
        # 内部仍然用官方 BinaryJaccardIndex
        self._jac = BinaryJaccardIndex(**kwargs)

        # 累加样本级 IoU 之和与样本计数
        self.add_state("sum_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_items", default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds  : [B, ...]  二值预测
            target : [B, ...]  二值标签
        """
        B = preds.shape[0]

        # 将每个样本展平到一维后单独喂入 BinaryJaccardIndex
        preds_flat  = preds.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        for i in range(B):
            score = self._jac(preds_flat[i], target_flat[i])  # 单样本 IoU
            self._jac.reset()  # 清掉内部状态，下一样本重新计算
            self.sum_iou += score
            self.n_items += 1

    def compute(self):
        # 返回所有样本 IoU 的平均值
        return self.sum_iou / self.n_items

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()

macro_dice_meter = Dice(average='samples').cuda()
macro_miou_meter = SampleMeanBinaryJaccard().cuda()

macro_dice_recoder = AverageMeter('Macro Dice', ':2.4f')
macro_miou_recoder = AverageMeter('Macro mIoU', ':2.4f')

micro_dice_meter = Dice().cuda()
micro_miou_meter = BinaryJaccardIndex().cuda()
micro_acc_meter = Accuracy(task='binary').cuda()

micro_acc_recoder = AverageMeter('Micro Acc', ':2.4f')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 512)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        # gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu()
        res = res.sigmoid()
        # res = res.numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res = (res * 255).astype(np.uint8)
        
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # th_lambda = 0.5
        # res[res >= int(255 * th_lambda)] = 255
        # res[res < int(255 * th_lambda)] = 0
        # print(type(gt), gt.shape, np.unique(gt))
        # print(type(res), res.shape, np.unique(res))
        # 1) 二值化到 {0,1}
        # gt_bin   = (gt  > 127).astype(np.uint8)
        # pred_bin = (res > 127).astype(np.uint8)

        # 2) 转成 torch 张量；允许是 (H,W) 或 (N,H,W)，torchmetrics 会自动展平额外维度
        # gt_t   = torch.from_numpy(gt_bin).cuda()
        # pred_t = torch.from_numpy(pred_bin).cuda()
        pred_t = res.squeeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0).cuda()
        gt_t = torch.floor_divide(gt_t, 255)  
        # gt_t = F.upsample(gt_t, size=(352, 352), mode='nearest').squeeze(0)
        # print(gt_t.shape)
        # print(pred_t.shape, gt_t.shape)
        # print(type(pred_t), type(gt_t))
        # print(pred_t.dtype, gt_t.dtype)
        
        macro_dice_meter.update(pred_t, gt_t.long())
        macro_miou_meter.update(pred_t, gt_t.long())
        # macro_dice_recoder.update(macro_dice_meter.compute().item())
        # macro_miou_recoder.update(macro_miou_meter.compute().item())
        # print(macro_dice_meter.compute().item())
        # print(macro_miou_meter.compute().item())
        
        micro_dice_meter.update(pred_t, gt_t.long())
        micro_miou_meter.update(pred_t, gt_t.long())
        micro_acc_meter.update(pred_t, gt_t.long())
        
        # micro_acc_recoder.update(micro_acc_meter.compute().item())

        print("Saving " + name)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        th_lambda = 0.5
        res[res >= int(255 * th_lambda)] = 255
        res[res < int(255 * th_lambda)] = 0
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
        # macro_dice_meter.reset()
        # macro_miou_meter.reset()
        # micro_acc_meter.reset()
        # break
    
# print(macro_dice_recoder.val)
# print(macro_miou_recoder.val)
print(macro_dice_meter.compute().item())
print(macro_miou_meter.compute().item())
print(micro_dice_meter.compute().item())
print(micro_miou_meter.compute().item())
# print(micro_acc_recoder.val)
print(micro_acc_meter.compute().item())

