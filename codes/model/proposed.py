from model import common

import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np

from model.transformer import Spa_TransformerEncoder

MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return ProposedModel(args)


class BasicModule(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, block_type='basic', bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        self.block_type = block_type

        m_body = []
        if block_type == 'basic':
            n_blocks = 10
            m_body = [
                common.BasicBlock(conv, n_feat, n_feat, kernel_size, bias=bias, bn=bn)
                # common.ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        elif block_type == 'residual':
            n_blocks = 5
            #n_blocks = 5
            m_body = [
                common.ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        else:
            print('Error: not support this type')
        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        res = self.body(x)
        if self.block_type == 'basic':
            out = res + x
        elif self.block_type == 'residual':
            out = res

        return out


class ProposedModel(nn.Module):

    def __init__(self, args, conv=common.default_conv):
        super(ProposedModel, self).__init__()

        self.args = args
        self.scale = args.scale[0]
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        # rgb_mean = (0.3972,0.4087,0.3683)    #AID

        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head body
        #n_feats=64, n_colors=3,kernel_size=3
        m_head = [
            conv(args.n_colors, n_feats, kernel_size),
        ]
        self.head = nn.Sequential(*m_head)

        # define main body
        self.feat_extrat_stage1 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)
        self.feat_extrat_stage2 = BasicModule(conv, n_feats, 5, block_type='residual', act=act)
        self.feat_extrat_stage3 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)


        reduction = 4
        self.stage1_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage2_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage3_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.up_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.span_conv1x1 = conv(n_feats // reduction, n_feats, 1)

        self.upsampler = common.Upsampler(conv, self.scale, n_feats, act=False)

        # define tail body
        # n_feats=64, n_colors=3,kernel_size=3
        self.tail = conv(n_feats, args.n_colors, kernel_size)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # define transformer
        image_size = args.patch_size // self.scale
        patch_size = 8
        dim = 512
        en_depth = args.en_depth
        de_depth = args.de_depth
        heads = 6
        mlp_dim = 512
        channels = n_feats // reduction
        dim_head = 32
        dropout = 0.0

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size
        self.patch_to_embedding_low1 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low2 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low3 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_high = nn.Linear(patch_dim, dim)

        self.embedding_to_patch = nn.Linear(dim, patch_dim)

        self.encoder_stage1 = Spa_TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage2 = Spa_TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage3 = Spa_TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_up = Spa_TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)


        #pixcel shuffle+conv-decoder
        self.decoder1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upConv=conv(16, self.scale*self.scale * 16,3)
        self.upShuffle=nn.PixelShuffle(self.scale)



    def forward(self, x):

        x = self.sub_mean(x)
        x = self.head(x)


        # feature extraction part
        feat_stage1 = self.feat_extrat_stage1(x)
        feat_stage2 = self.feat_extrat_stage2(feat_stage1)
        feat_stage3 = self.feat_extrat_stage3(feat_stage2)
        feat_ups = self.upsampler(feat_stage3)



        feat_stage1 = self.stage1_conv1x1(feat_stage1)
        feat_stage2 = self.stage2_conv1x1(feat_stage2)
        feat_stage3 = self.stage3_conv1x1(feat_stage3)
        feat_ups = self.up_conv1x1(feat_ups)


        # transformer part:
        p = self.patch_size

        feat_stage1 = rearrange(feat_stage1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        feat_stage2 = rearrange(feat_stage2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage3 = rearrange(feat_stage3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_ups = rearrange(feat_ups, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)


        feat_stage1 = self.patch_to_embedding_low1(feat_stage1)
        feat_stage2 = self.patch_to_embedding_low2(feat_stage2)
        feat_stage3 = self.patch_to_embedding_low3(feat_stage3)
        feat_ups = self.patch_to_embedding_high(feat_ups)


        # encoder
        feat_stage1, feat_stage1_sparse = self.encoder_stage1(feat_stage1)
        feat_stage2, feat_stage2_sparse = self.encoder_stage2(feat_stage2)
        feat_stage3, feat_stage3_sparse = self.encoder_stage3(feat_stage3)
        feat_ups, feat_ups_sparse = self.encoder_up(feat_ups)

        loss_sparse = (feat_stage1_sparse + feat_stage2_sparse + feat_stage3_sparse + feat_ups_sparse) / 4
        # print('loss_sparse-all--------',loss_sparse)





        # pixcel_shuffle+conv->decoder
        feat_ups = self.embedding_to_patch(feat_ups)
        feat_ups = rearrange(feat_ups, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.args.patch_size // p, p1=p, p2=p)
        feat_ups = self.span_conv1x1(feat_ups)

        feat_stage3 = self.embedding_to_patch(feat_stage3)
        feat_stage3 = rearrange(feat_stage3, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.args.patch_size // (self.scale*p), p1=p,p2=p)
        feat_stage3 = self.upConv(feat_stage3)
        feat_stage3 = self.upShuffle(feat_stage3)
        feat_stage3 = self.span_conv1x1(feat_stage3)

        feat_stage2 = self.embedding_to_patch(feat_stage2)
        feat_stage2 = rearrange(feat_stage2, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.args.patch_size // (self.scale*p), p1=p, p2=p)
        feat_stage2 = self.upConv(feat_stage2)
        feat_stage2 = self.upShuffle(feat_stage2)
        feat_stage2 = self.span_conv1x1(feat_stage2)

        feat_stage1 = self.embedding_to_patch(feat_stage1)
        feat_stage1 = rearrange(feat_stage1, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.args.patch_size // (self.scale*p), p1=p, p2=p)
        feat_stage1 = self.upConv(feat_stage1)
        feat_stage1 = self.upShuffle(feat_stage1)
        feat_stage1 = self.span_conv1x1(feat_stage1)



        feat_ups = self.decoder3(feat_ups + feat_stage3)
        feat_ups = self.decoder2(feat_ups + feat_stage2)
        feat_ups = self.decoder1(feat_ups + feat_stage1)


        x = self.tail(feat_ups)
        x = self.add_mean(x)

        
        return x,loss_sparse


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    from option import args
    model = TransENet(args)
    model.eval()
    input = torch.rand(1, 3, 48, 48)
    sr = model(input)
    #print(sr.size())
