from torch import nn
import torch.nn.functional as F


CONV_TYPE = {0: 'conv_1', 1: 'conv_3', 2: 'conv_5'}
NORM_TYPE = {0: None, 1: 'bn', 2: 'in'}
UP_TYPE = {0: 'bilinear', 1: 'nearest', 2: 'deconv'}
SHORT_CUT_TYPE = {0: False, 1: True}
SKIP_TYPE = {0: False, 1: True}


def decimal2binary(n):
    return bin(n).replace("0b", "")


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3):
        super(Conv, self).__init__()
        self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.norm_type = None
        self.conv_op = nn.Conv2d(C_in, C_out, kernel_size=self.kernel_size, padding=(self.kernel_size - 1)//2)
        self.sep_conv = nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=self.kernel_size, padding=(self.kernel_size - 1)//2, groups=C_in),
                                      nn.Conv2d(C_in, C_out, kernel_size=1,stride=1,padding=0))
        self.bn = nn.BatchNorm2d(C_out)
        self.inn = nn.InstanceNorm2d(C_out)

    def set_arch(self, conv_id, norm_id):
        self.conv_id = conv_id
        self.conv_type = CONV_TYPE[conv_id]
        self.norm_type = NORM_TYPE[norm_id]
        # self.kernel_size = kernel_size

    def forward(self, x):
        if self.conv_id <=2 :
            if self.conv_type == 'conv_1':
                self.kernel_size = 1
            elif self.conv_type == 'conv_3':
                self.kernel_size = 3
            elif self.conv_type == 'conv_5':
                self.kernel_size = 5
            h = self.conv_op(x)
        else:
            if self.conv_type == 'sep_conv3':
                self.kernel_size = 3
            elif self.conv_type == 'sep_conv5':
                self.kernel_size = 5
            h = self.sep_conv(x)

        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(h)
            elif self.norm_type == 'in':
                h = self.inn(h)

        h = self.activation(h)
        return h

class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, num_skip_in):
        super(Cell, self).__init__()
        #
        # self.post_conv1 = PostGenBlock(in_channels, out_channels, ksize=ksize, up_block=True)
        # self.pre_conv1 = PreGenBlock(in_channels, out_channels, ksize=ksize, up_block=True)
        #
        # self.post_conv2 = PostGenBlock(out_channels, out_channels, ksize=ksize, up_block=False)
        # self.pre_conv2 = PreGenBlock(out_channels, out_channels, ksize=ksize, up_block=False)
        # self.cur_cell = cur_cell
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)
        self.deconv_sc = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # skip_in
        self.skip_deconvx2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_deconvx4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

        self.num_skip_in = num_skip_in
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])
        # self.skip_in_op = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def set_arch(self, up_id, conv_id, norm_id, short_cut_id, skip_ins=None):
        # self.post_conv1.set_arch(up_id, norm_id)
        # self.pre_conv1.set_arch(up_id, norm_id)
        # self.post_conv2.set_arch(up_id, norm_id)
        # self.pre_conv2.set_arch(up_id, norm_id)

        self.conv1.set_arch(conv_id, norm_id)
        self.conv2.set_arch(conv_id, norm_id)
        self.skip_ins = []
        for i in range(self.num_skip_in):
            self.skip_ins.append(UP_TYPE[skip_ins[i]])

        # if self.num_skip_in == 1:
        #     # self.skip_ins = [0 for _ in range(self.num_skip_in)]
        #     # for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
        #     #     self.skip_ins[-(skip_idx + 1)] = int(skip_in)
        #     self.skip_ins = [UP_TYPE[skip_ins1]]
        # elif self.num_skip_in == 2:
        #     self.skip_ins =[UP_TYPE[skip_ins1], UP_TYPE[skip_ins2]]
            # self.skip_ins2 = UP_TYPE[skip_ins2]

        self.up_type = UP_TYPE[up_id]
        self.short_cut = UP_TYPE[short_cut_id]

    def forward(self, x, skip_ft=None):
        residual = x

        # Up_sample
        # print(x.shape)
        if self.up_type == 'deconv':
            h = self.deconv_sc(x)
        else:
            h = F.interpolate(x, scale_factor=2, mode=self.up_type)

        # print("cell_up_shape:{}".format(h.shape))
        _, _, ht, wt = h.size()

        # first conv
        h = self.conv1(h)
        # print("cell_conv1_shape:{}".format(h.shape))
        h_skip_out = h

        # second conv
        if skip_ft:
            for i,ft in enumerate(skip_ft):
                if self.skip_ins[i] != 'deconv':
                    h += self.skip_in_ops[i](F.interpolate(ft, size=(ht, wt), mode=self.skip_ins[i]))
                else:
                    scale = wt // ft.size()[-1]
                    h += self.skip_in_ops[i](getattr(self, f'skip_deconvx{scale}')(ft))

        final_out = self.conv2(h)
        # print("cell_conv2_shape:{}".format(final_out.shape))

        # shortcut

        if self.short_cut != 'deconv':
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.short_cut))
        else:
            final_out += self.c_sc(self.deconv_sc(x))

        # print("cell_conv2_shape:{}".format(final_out.shape))

        return h_skip_out, final_out


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)