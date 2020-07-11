import torch
from torch import nn
import torch.nn.functional as F


class CEN(nn.Module):
    def __init__(self):
        super(CEN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=35),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1),
            nn.ConvTranspose2d(2, 2, kernel_size=16, stride = 8, padding=0),
            )

    def forward(self, x):
        (b, c, w, h) = x.shape
        x = self.net(x)
        x = F.softmax(x, dim=1)
        top_left_x, top_left_y = (x.shape[2]-w)//2, (x.shape[3]-h)//2
        return x[:,:,top_left_x:top_left_x+w, top_left_y:top_left_y+h]


class PCN(nn.Module):
	def __init__(self):
		super(PCN, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Flatten(),
			nn.Linear(in_features=4608, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=512),
			)

	def forward(self, x):
		x = self.net(x)
		return x


class CMN(nn.Module):
    def __init__(self):
        super(CMN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2, groups=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1, groups=2),  # (b x 384 x 13 x 13)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=13, stride=1),  # (b x 256 x 6 x 6)
            )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return x
