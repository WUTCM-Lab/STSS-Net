import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import functools
from einops import rearrange, repeat
import models
from models.help_funcs import Transformer, TransformerDecoder, mlp_head, tokenizer_se_block, SpaceCenter_Concentrate_Attention


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
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
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


# 主函数
def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[], input_size=198):
    if args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(args=args, token_len=4, with_pos='learned', input_size=input_size)

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(args=args, token_len=4, with_pos='learned', enc_depth=1, dec_depth=8, input_size=input_size)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(args=args, with_pos='learned', enc_depth=1, dec_depth=4,
                               decoder_dim_head=8, input_size=input_size)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################
class BASE_Transformer(torch.nn.Module):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, args, with_pos, token_len=8, token_trans=True, enc_depth=1, dec_depth=1, dim_head=64,
                 decoder_dim_head=64, tokenizer=True, with_cross=True, pool_mode='max', pool_size=2,
                 decoder_softmax=True, with_decoder_pos=None, with_decoder=True, input_size=198):
        super(BASE_Transformer, self).__init__()

        # 开关
        # encoder参数开关
        self.with_pos = with_pos
        # cross attention开关
        self.try_cross = with_cross
        # decoder参数开关
        self.with_decoder_pos = with_decoder_pos
        # token转换器开关
        self.tokenizer = tokenizer
        # encoder开关
        self.token_trans = token_trans
        # decoder开关
        self.with_decoder = with_decoder

        # 参数定义
        # 标记器转换token长度
        self.token_len = token_len
        # patch 大小
        self.patches = args.patches
        # transform长度及头长度
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head

        dim = 64
        mlp_dim = dim * 2

        if self.with_pos == 'learned':  # en可学习参数
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 64))
            self.s_pos_embedding = nn.Parameter(torch.randn(1, 64, self.token_len * 2))

        if self.with_decoder_pos == 'learned':  # de可学习参数
            decoder_pos_size = 3  # 这里需要重新定义？？？？
            if self.try_cross:
                self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 64,
                                                                      decoder_pos_size,
                                                                      decoder_pos_size))
            else:
                self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.token_len + 1, 64))
                self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        if not self.tokenizer:  # 无标记器时自适应池化层定义参数
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        # 网络结构定义
        self.SpaceCenter_Concentrate_Attention = SpaceCenter_Concentrate_Attention(args=args)  # 空间去冗块
        # self.SpaceCenter_Concentrate_Attention2 = SpaceCenter_Concentrate_Attention(args=args)
        self.conv = nn.Sequential(  # 卷积块
            nn.Conv2d(input_size, 128, kernel_size=1, bias=False),  # [b, 198, 9, 9] - [b, 128, 9, 9]
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # [b, 128, 9, 9] - [b, 128, 9, 9]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, bias=False),  # [b, 96, 9, 9] - [b, 64, 9, 9]
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # [b, 64, 9, 9] - [b, 64, 9, 9]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_a = nn.Conv2d(64, self.token_len, kernel_size=1, padding=0, bias=False)  # 空间标记器的卷积层
        self.tokenizer_se_block = tokenizer_se_block(channels=64, args=args)  # 新标记器定义
        self.sequential_transformer = Transformer(dim=16, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth, heads=8,
                                                      dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=decoder_softmax)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 有cross attention时要池化到一维数据
        self.to_latent = nn.Identity()  # 无cross时对数据的处理
        self.classifier = mlp_head(dim=64)  # 分类器
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64, False),
            nn.ReLU(),
        )

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)  # Shape [8, 32, 3 , 3] - Shape [8, self.token_len, 3, 3]
        spatial_attention = spatial_attention.view(
            [b, self.token_len, -1]).contiguous()  # Shape [8, self.token_len, 3 , 3] - Shape [8, self.token_len, 9]
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  # Shape [8, 32, 3 , 3] - Shape [8, 32, 9]

        tokens = torch.einsum('bln,bcn->blc', spatial_attention,
                              x)  # Shape [8, self.token_len, 9] - Shape [8, self.token_len, 32]

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size*2])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size*2])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_sequential_transformer(self, x):
        x = rearrange(x, 'b l c -> b c l')
        if self.with_pos:
            x += self.s_pos_embedding
        x = self.sequential_transformer(x)
        x = rearrange(x, 'b c l -> b l c')
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'learned':
            if self.try_cross:
                x = x + self.pos_embedding_decoder
            else:
                b, t, _ = m.shape
                cls_tokens = repeat(self.cls_token, '() t d -> b t d', b=b)
                m = torch.cat((cls_tokens, m), dim=1)
                m += self.pos_embedding_decoder
                m = self.transformer_decoder(x, m)
                return m
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        if not self.patches == 1:
            x1, att = self.SpaceCenter_Concentrate_Attention(x1)  # Shape [b, band, 9, 9] - Shape [b, band, 9, 9]
            x2, att = self.SpaceCenter_Concentrate_Attention(x2)
            x1 = self.conv(x1)  # Shape [b, band, 9, 9] - Shape [b, 64, 9, 9]
            x2 = self.conv(x2)
        else:
            x1 = self.fc(x1.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
            x2 = self.fc(x2.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)

        #  forward tokenzier
        if self.tokenizer:
            # token1 = self._forward_semantic_tokens(x1) # Shape [b, 32, 3 , 3] - Shape [b, 4, 32]
            # token2 = self._forward_semantic_tokens(x2)
            token1 = self.tokenizer_se_block(x1)  # Shape [b, 64, 9, 9] - Shape [b, 8, 64]
            token2 = self.tokenizer_se_block(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)

        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)  # [b, 16, 64] - [b, 16, 64]
            self.tokens = self._forward_sequential_transformer(self.tokens)  # [b, 16, 64] - [b, 16, 64]
            token1, token2 = self.tokens.chunk(2, dim=1)

        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)  # [b, 81, 64] + [b, 8, 64] - Shape [b, 64, 9, 9]
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        if self.try_cross:
            # feature differencing
            x = torch.abs(x1 - x2)
            # 将特征图输入到平均池化层中进行平坦化
            x = self.avg_pool(x)  # [8,64,p,p] -> Shape [8, 64, 1, 1]
            # 将结果展平为一维向量
            x = x.view(x.size(0), -1)  # Shape [8, 32]
        else:
            # feature differencing
            x1 = self.to_latent(x1[:, 0])
            x2 = self.to_latent(x2[:, 0])
            x = torch.abs(x1 - x2)

        x = self.classifier(x)
        return x


