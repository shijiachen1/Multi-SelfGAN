from torch import nn
from models.building_blocks import Cell, DisBlock, OptimizedDisBlock

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.tea_gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.tea_gf_dim)
        self.cell1 = Cell(args.tea_gf_dim, args.tea_gf_dim, 'nearest', num_skip_in=0, short_cut=True)
        self.cell2 = Cell(args.tea_gf_dim, args.tea_gf_dim, 'bilinear', num_skip_in=0, short_cut=False)
        self.cell3 = Cell(args.tea_gf_dim, args.tea_gf_dim, 'bilinear', num_skip_in=0, short_cut=False, norm='bn')
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.tea_gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.tea_gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        tea_img_1 = self.to_rgb(h1)
        h2_skip_out, h2 = self.cell2(h1)
        tea_img_2 = self.to_rgb(h2)
        _, h3 = self.cell3(h2)
        output = self.to_rgb(h3)

        return tea_img_1, tea_img_2, output

class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
