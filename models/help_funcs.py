import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import math


class mlp_head(nn.Sequential):
    def __init__(self, dim):
        super().__init__(nn.LayerNorm(dim),
                         nn.Linear(dim, 2)
                         )


class SpaceCenter_Concentrate_Attention(torch.nn.Module):
    def __init__(self, args, weight=0.5):
        super(SpaceCenter_Concentrate_Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.patches = args.patches
        self.batch_size = args.batch_size
        self.linear = nn.Linear(self.patches*self.patches, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.patches*self.patches, self.patches*self.patches // 2, False),
            nn.ReLU(),
            nn.Linear(self.patches*self.patches // 2, 1, False),
            # nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.mask = torch.zeros([self.patches, self.patches]).cuda()
        self.cal_mask()
        self.weight = weight

    def cal_mask(self):
        for x in range(0, self.patches):
            for y in range(0, self.patches):
                len = math.sqrt((x + 0.5 - self.patches / 2) ** 2 + (y + 0.5 - self.patches / 2) ** 2)
                self.mask[x][y] = len

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        feature_map = torch.cat([avgout, maxout], dim=1)
        feature_map = self.conv(feature_map)
        feature_map = self.sigmoid(feature_map)
        b, c, h, w = feature_map.size()

        para_b = feature_map.view(b, -1)[:, (self.patches*self.patches)//2]
        para_k = self.fc(feature_map.view(b, -1))
        para_k = self.Softplus(para_k)*(-1)  # k<0
        print(para_k)
        attention_map = torch.zeros([b, 1, self.patches, self.patches]).cuda()

        for i in range(0, self.patches):
            for j in range(0, self.patches):
                attention_map[:, 0, i, j] = self.mask[i][j] * para_k.squeeze(dim=-1) + para_b
        attention_map = self.sigmoid(attention_map)

        mask = torch.zeros([b, 1, self.patches, self.patches]).cuda()
        mask += para_b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cut_map = abs(feature_map - mask)
        cut_map = 1-cut_map

        attention_map = self.weight * attention_map + (1 - self.weight) * cut_map

        return attention_map * x, attention_map


class tokenizer_se_block(nn.Module):
    def __init__(self, channels, args, ratio=4):
        super(tokenizer_se_block, self).__init__()

        # self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.sigmoid = nn.Sigmoid()
        self.patches = args.patches

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels // ratio // 2, False),
            # nn.Sigmoid()
        )

        # 经过两次全连接层，学习不同通道的重要性
        self.fc_center = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels // ratio // 2, False),
            # nn.Sigmoid()
        )

        self.fc_context = nn.Sequential(
            nn.Linear((channels*(self.patches*self.patches-1)), (channels*(self.patches*self.patches-1)) // ratio, False),
            nn.ReLU(),
            nn.Linear((channels*(self.patches*self.patches-1)) // ratio, (channels*(self.patches*self.patches-1)) // ratio // 2, False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        if not self.patches == 1:
            b, c, h, w = x.size()  # 取出batch size和通道数
            # b,c,w,h->b,c,3,3->b,c,9 压缩与通道信息学习
            # avg = self.avgpool(x).view(b, c, 9)
            avg = x.view(b, c, h * w)

            center = avg[:, :, h * w // 2]
            background = torch.cat((avg[:, :, :h * w // 2], avg[:, :, h * w // 2 + 1:]), dim=-1).contiguous()
            avg = torch.cat((center.unsqueeze(-1), background), dim=-1)
            # background = rearrange(background, 'b c l -> b (l c)').contiguous()
            background = rearrange(background, 'b c l -> b l c')

            # b,c->b,c/ratio/ratio->b,c,1,1 激励操作
            center = self.fc_center(center)
            b, token_len = center.size()
            # background = self.fc_context(background)
            background = self.fc(background)
            # background = rearrange(background.view(b, h*w-1, token_len), 'b l c -> b c l')
            background = rearrange(background, 'b l c -> b c l')

            attention_map = torch.cat((center.view(b, token_len, 1), background), dim=-1)
            attention_map = torch.softmax(attention_map, dim=-2)

            tokens = torch.einsum('bln,bcn->blc', attention_map, avg)
        else:
            center = x
            # b,c->b,c/ratio/ratio->b,c,1,1 激励操作
            center = self.fc_center(center.squeeze(-1).squeeze(-1))
            attention_map = torch.softmax(center, dim=-1)
            tokens = torch.einsum('bln,bcn->blc', attention_map.unsqueeze(-1), x.squeeze(-1))

        return tokens


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, softmax=softmax))),
                # Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
            # m = attn(m, mask=mask)
            # m = ff(m)
        return x


