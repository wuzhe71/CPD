import numpy as np
import torch


class cal_maxF(object):
    # max Fmeasure
    def __init__(self, num, thds=256):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
        self.idx += 1

    def cal(self, pred, gt):
        pred = np.uint8(pred*255)
        target = torch.from_numpy(pred[gt > 0.5])
        nontarget = torch.from_numpy(pred[gt <= 0.5])
        targetHist = torch.histc(target.float(), 256, 0, 255)
        nontargetHist = torch.histc(nontarget.float(), 256, 0, 255)
        targetHist = self.flip(targetHist).cumsum(dim=0)
        nontargetHist = self.flip(nontargetHist).cumsum(dim=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist.numpy() / np.sum(gt)

        return precision.numpy(), recall

    def flip(self, tensor):
        inv_idx = torch.arange(tensor.size(0) - 1, -1, -1).long()
        return tensor[inv_idx]

    def show(self):
        assert self.num == self.idx
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)

        return fmeasure.max()


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        return np.mean(np.abs(pred-gt))

    def show(self):
        return np.mean(self.prediction)