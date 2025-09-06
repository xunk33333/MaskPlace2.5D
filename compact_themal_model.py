#themal model from
#ATPlace2.5D: Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC
import torch
import torch.nn as nn

def F(a, b, c):
    delta = torch.sqrt(a**2 + b**2 + c**2)
    term1 = b * torch.log((c + delta) / torch.sqrt(a**2 + b**2))
    term2 = c * torch.log((b + delta) / torch.sqrt(a**2 + c**2))
    term3 = a * torch.atan((b * c) / (a * delta))
    return (2 / torch.sqrt(torch.tensor(torch.pi))) * (term1 + term2 - term3)

class ChipletThermalModel(nn.Module):
    def __init__(self, num_chiplets):
        super().__init__()
        self.N = num_chiplets

        # 可训练参数
        self.A = nn.Parameter(torch.ones(1))               # 全局振幅因子
        self.a = nn.Parameter(torch.ones(1))               # 全局厚度因子
        self.B = nn.Parameter(torch.zeros(1))              # 偏置项

        self.lx = nn.Parameter(torch.ones(self.N))         # 每个chiplet的lx
        self.ly = nn.Parameter(torch.ones(self.N))         # 每个chiplet的ly

    def forward(self, x, y, chiplets_x, chiplets_y, chiplets_width, chiplets_height, chiplets_power, grid, train=True):
        """
        x, y: shape (B,) - 坐标位置
        chiplets: dict with tensors of shape (N,) for:
            - xi, yi: chiplet 中心坐标
            - wi, hi: chiplet 尺寸
            - Pi: chiplet 功率
        """
        result = torch.zeros_like(x)
        if train:
            TRAIN = x.shape[0]
        else:
            TRAIN = 1
        for i in range(self.N):
            dx = x - chiplets_x[:,i].view(TRAIN,-1).repeat(1, grid*grid)
            dy = y - chiplets_y[:,i].view(TRAIN,-1).repeat(1, grid*grid)

            w2 = chiplets_width[:,i].view(TRAIN,-1).repeat(1, grid*grid) / 2
            h2 = chiplets_height[:,i].view(TRAIN,-1).repeat(1, grid*grid) / 2
            Pi = chiplets_power[:,i].view(TRAIN,-1).repeat(1, grid*grid)

            # 四项热扩散
            terms = []
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    b = ( w2 - sx * dx) / self.lx[i]
                    c = ( h2 - sy * dy) / self.ly[i]
                    terms.append(F(self.a, b, c))

            sumF = sum(terms) + self.B
            result += Pi * self.A * sumF

        return result 
