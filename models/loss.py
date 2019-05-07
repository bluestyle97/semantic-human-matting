import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionL1Loss(nn.Module):
    def __init__(self, gamma=0.5):
        super(PredictionL1Loss, self).__init__()
        self.gamma = gamma

    def __call__(self, img, alpha_p, alpha_g):
        l_alpha = F.l1_loss(alpha_p, alpha_g)

        fg_p = alpha_p.repeat(1, 3, 1, 1) * img
        fg_g = alpha_g.repeat(1, 3, 1, 1) * img
        l_comps = F.l1_loss(fg_p, fg_g)

        l_p = self.gamma * l_alpha + (1 - self.gamma) * l_comps
        return l_p, l_alpha, l_comps


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def __call__(self, trimap_p, trimap_g):
        n, _, h, w = trimap_p.size()
        l_t = F.cross_entropy(trimap_p, trimap_g.view(n, h, w).long())
        return l_t


if __name__ == '__main__':
    img = torch.rand(4, 3, 256, 256)
    alpha_p = torch.rand(4, 1, 256, 256)
    alpha_g = torch.rand(4, 1, 256, 256)

    trimap_p = torch.rand(4, 3, 256, 256)
    trimap_g = torch.randint(3, (4, 256, 256), dtype=torch.long)

    Lp = PredictionL1Loss()
    Lc = ClassificationLoss()

    lp = Lp(img, alpha_p, alpha_g)
    print(lp.size(), lp)
    lc = Lc(trimap_p, trimap_g)
    print(lc.size(), lc)