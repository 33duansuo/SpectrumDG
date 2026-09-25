"""物理先验模块 (非侵入式, 与 RadioWNet 解耦)。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalMutationAttention(nn.Module):
    """Sobel 提取建筑边缘 (绕射突变区), 生成 loss 权重图。
    边缘区域权重高, 建筑内部权重 0, 开阔区域权重 1。"""

    def __init__(self, edge_boost=1.5, base=1.0):
        super().__init__()
        kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sx', kx)
        self.register_buffer('sy', ky)
        self.edge_boost, self.base = edge_boost, base

    def forward(self, building):
        b = building / (building.amax(dim=(2, 3), keepdim=True) + 1e-6)
        gx = F.conv2d(b, self.sx, padding=1)
        gy = F.conv2d(b, self.sy, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        interior = (b > 0.5).float()
        return (self.base + self.edge_boost * edge) * (1.0 - interior)


class PhysicsElection(nn.Module):
    """MC-Dropout 生成 N 个候选 radio map, 按物理规律评分后 softmax 加权选出最佳。
    非侵入式: 用 forward hook 钩在 Wlayer5 瓶颈处, 不改 RadioWNet 层定义。"""

    def __init__(self, rwnet, n=5, p=0.1, temp=1.0):
        super().__init__()
        self.rwnet = rwnet
        self.n, self.p, self.temp = n, p, temp
        self._inject = False
        self.rwnet.Wlayer5.register_forward_hook(self._hook)
        kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sx', kx)
        self.register_buffer('sy', ky)

    def _hook(self, m, i, o):
        return F.dropout2d(o, p=self.p, training=True) if self._inject else o

    def _score(self, pred, b, tx):
        # 1. 值域范围
        s_range = -(F.relu(-pred) + F.relu(pred - 1)).mean(dim=(2, 3))
        # 2. 距离单调性
        H, W = pred.shape[-2:]
        yy = torch.arange(H, device=pred.device).view(1, 1, H, 1).float()
        xx = torch.arange(W, device=pred.device).view(1, 1, 1, W).float()
        ts = tx.sum(dim=(2, 3), keepdim=True) + 1e-8
        cy = (tx * yy).sum(dim=(2, 3), keepdim=True) / ts
        cx = (tx * xx).sum(dim=(2, 3), keepdim=True) / ts
        D = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        D = D / (D.amax(dim=(2, 3), keepdim=True) + 1e-6)
        muD, muP = D.mean(dim=(2, 3), keepdim=True), pred.mean(dim=(2, 3), keepdim=True)
        s_mono = -((D - muD) * (pred - muP)).mean(dim=(2, 3))
        # 3. 边缘一致性
        gx = F.conv2d(pred, self.sx, padding=1)
        gy = F.conv2d(pred, self.sy, padding=1)
        bx = F.conv2d(b, self.sx, padding=1)
        by = F.conv2d(b, self.sy, padding=1)
        s_edge = (torch.sqrt(gx * gx + gy * gy + 1e-8)
                  * torch.sqrt(bx * bx + by * by + 1e-8)).mean(dim=(2, 3))
        return (s_range + s_mono + s_edge).squeeze(1)

    def forward(self, inp):
        self._inject = True
        outs = [self.rwnet(inp) for _ in range(self.n)]
        self._inject = False
        out1 = outs[0][0]
        cands = torch.stack([o[1] for o in outs], 0)
        b, tx = inp[:, 0:1], inp[:, 1:2]
        sc = torch.stack([self._score(cands[i], b, tx) for i in range(self.n)], 0)
        w = F.softmax(sc / self.temp, dim=0).view(self.n, -1, 1, 1, 1)
        return [out1, (w * cands).sum(0)]
