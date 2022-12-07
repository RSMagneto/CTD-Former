import functools
import torch
from torch import nn, softmax
from torch.nn import init, Conv2d
from torch.optim import lr_scheduler

import models

from einops import rearrange, repeat
from models.help_funcs import TwoLayerConv2d, TransformerDecoder, Cross_Attention, Residual2, PreNorm2, \
    Residual, PreNorm, FeedForward, Attention, Attention1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_scheduler(optimizer, args):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = models.ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'CTD-Former':
        net = Model( )

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class CrossTransformer(nn.Module):
    """
  Cross Transformer layer
  """

    def __init__(self, dim=32,inner_dim=512, depth=1, heads=8, dim_head=64, mlp_dim=64, dropout=0):

        super(CrossTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim,Cross_Attention(dim, heads=heads,
                                                        dim_head=dim_head, dropout=dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x1, x2, mask=None):
        for attn, ff in self.layers:
            x1 = attn(x1, x2, mask=mask)
            x1 = ff(x1)
        return x1


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out

class DownsampleSR(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsampleSR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out

class Conv6432(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv6432, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out

class Convde6432(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convde6432, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out

class Conv2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=28),

        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):

        out = self.conv(x)
        return out

class PatchEmbeding(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim,  pool = 'cls', channels = 32, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        in_channels = 32
        out_channels = 32
        self.to_patch_embedding =  Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # self.avgpool = nn.AvgPool2d(kernel_size=16, stride=16, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = self.maxpool(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DilatedResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, d_rate=[1, 1]):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=d_rate[0], dilation=d_rate[0],
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride
        if inplanes == planes:
            self.identity = None
        else:
            self.identity = conv1x1(inplanes, planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.identity is not None:
            identity = self.identity(x)
        out += identity
        out = self.relu(out)
        return out


class DilatedResBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, d_rate=[5, 3, 1]):
        super(DilatedResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(64, planes, kernel_size=3, padding=1,stride=stride, bias=False)
        self.dconv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=3, dilation=d_rate[1], bias=False)
        self.dconv2 = nn.Conv2d(64, planes, kernel_size=3, stride=stride, padding=3, dilation=d_rate[1], bias=False)
        self.avgpooling = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.relu(x2)
        x2 = self.conv2(x2)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dconv1(out)
        out = self.relu(out)
        out = self.dconv2(out)
        output2 = out+x2
        return output2


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class Model(nn.Module):
    def __init__(self,output_sigmoid=False,output_nc=2):
        super(Model,self).__init__()
        self.EModel = EModel()
        self.SRModel = SRModel()
        self.CrossModel = CrossModel()
        self.DEModel =DEModel()
        self.classifier = TwoLayerConv2d(in_channels=256, out_channels=output_nc)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.EXModel = EXModel()
    def forward(self, x1=None,x2=None):
          x1=self.EXModel(x1)
          E1 = self.EModel(x1)
          SR1 = self.SRModel(x1)
          x2 = self.EXModel(x2)
          E2 = self.EModel(x2)
          SR2 = self.SRModel(x2)
          c1 = self.CrossModel(x1,x2)
          c2 = self.CrossModel(x2,x1)
          DE1 = self.DEModel(E1,c1)
          DE2 = self.DEModel(E2, c2)
          output1 = torch.cat((E1,DE1), dim=1)
          output1 = torch.cat((output1,SR1), dim=1)
          output2 = torch.cat((E2, DE2), dim=1)
          output2 = torch.cat((output2,SR2), dim=1)
          output = torch.cat((output1,output2), dim=1)
          output = self.upsamplex4(output)
          output = self.classifier(output)
          if self.output_sigmoid:
             output = self.sigmoid(output)

          return output


class EXModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=32,
                 resnet_stages_num=4, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):

        super(EXModel,self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=256, out_channels=output_nc)
        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_4 = self.resnet.layer1(x)
        x_8 = self.resnet.layer2(x_4)
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError
        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        x = self.conv_pred(x)
        return x


class EModel(nn.Module):
    def __init__(self,dim=32):
        super(EModel,self).__init__()
        self.downsample = Downsample(dim, dim)
    def forward(self, x):
        x = self.downsample(x)
        return x

class SRModel(nn.Module):
    def __init__(self):
        super(SRModel,self).__init__()
        self.SR2 = DilatedResBlock2(32, 64)
        self.conv6432 = Conv6432(64, 32)
        self.convde6432 = Convde6432(32, 64)
        self.Deconv = DeformConv2d(32, 32)
    def forward(self, x):
        SR = self.SR2(x)
        SR = self.conv6432(SR)
        SR = self.Deconv(SR)
        SR = SR + x
        SR = self.convde6432(SR)
        return SR

class CrossModel(nn.Module):
    def __init__(self):
        super(CrossModel,self).__init__()
        self.transformer = nn.ModuleList(
            [CrossTransformer(dim=32, depth=1, heads=8, dim_head=64, mlp_dim=64, dropout=0) for i in range(1)])
        self.patchembeding = PatchEmbeding(image_size=128, patch_size=8, num_classes=1000, dim=32, pool='cls',
                                                channels=32, emb_dropout=0.)
    def forward(self,x1,x2):
        output1 = self.patchembeding(x1)
        output2 = self.patchembeding(x2)
        for l in self.transformer:
            output= l(output1, output2)

        return output

class DEModel(nn.Module):
    def __init__(self, decoder_softmax=True, with_decoder_pos=None):
        super(DEModel, self).__init__()
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.transformer_decoder=TransformerDecoder(dim=32, depth=1, heads=8, dim_head=64, mlp_dim=32, dropout=0,
                                                      softmax=decoder_softmax)
    def forward(self,x1,x2):
        b, c, h, w = x1.shape
        if self.with_decoder_pos == 'fix':
            x1 = x1 + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x1 = x1 + self.pos_embedding_decoder
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x1, x2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)

        return x

if __name__ =='__main__':
    Model()