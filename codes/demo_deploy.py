from option import args
import model
import utils
import data.common as common

import torch
import numpy as np
import os
import glob
import cv2

# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
# -- coding: utf-8 --
import torchvision
from ptflops import get_model_complexity_info


import time




device = torch.device('cpu' if args.cpu else 'cuda')


def deploy(args, sr_model):

    img_ext = '.png'
    # img_ext = '.tif'
    img_lists = glob.glob(os.path.join(args.dir_data, '*'+img_ext))

    if len(img_lists) == 0:
        print("Error: there are no images in given folder!")

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with torch.no_grad():
        for i in range(len(img_lists)):
            print("[%d/%d] %s" % (i+1, len(img_lists), img_lists[i]))
            lr_np = cv2.imread(img_lists[i], cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)

            if args.cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * args.scale[0], lr_np.shape[1] * args.scale[0]),
                                interpolation=cv2.INTER_CUBIC)

            lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0)

            if args.test_block:
                # test block-by-block

                b, c, h, w = lr.shape
                # print(lr.shape)
                # factor = args.scale[0]
                factor = args.scale[0] if not args.cubic_input else 1

                tp = args.patch_size
                if not args.cubic_input:
                    ip = tp // factor
                else:
                    ip = tp


                assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                if not args.cubic_input:
                    sr = torch.zeros((b, c, h * factor, w * factor))
                else:
                    sr = torch.zeros((b, c, h, w))

                for iy in range(0, h, ip):

                    if iy + ip > h:
                        iy = h - ip
                        # print('h', h, ' w', w, ' ip', ip)
                    ty = factor * iy

                    for ix in range(0, w, ip):
                        # print('ix',ix,'ip',ip,'w',w)
                        # print(ix + ip > w)
                        if ix + ip > w:
                            ix = w - ip
                            # print('ix',ix)
                        tx = factor * ix

                        # forward-pass
                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        # print(lr_p.shape,tx,tx + tp,ty,ty + tp)
                        lr_p = lr_p.to(device)


                        if args.loss == '1*L1_SparseN':
                            sr_p, loss_p = sr_model(lr_p)
                        elif args.loss == '1*L1_KL':
                            sr_p, encodeResult_p, encodeResult_3p, encodeResult_2p, encodeResult_1p = sr_model(lr_p)
                        else:
                            sr_p = sr_model(lr_p)
                        # print(sr_p.size(),sr.size())

                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p

            else:

                lr = lr.to(device)
                sr = sr_model(lr)


            sr_np = np.array(sr.cpu().detach())
            sr_np = sr_np[0, :].transpose([1, 2, 0])
            lr_np = lr_np * args.rgb_range / 255.

            # Again back projection for the final fused result
            for bp_iter in range(args.back_projection_iters):
                sr_np = utils.back_projection(sr_np, lr_np, down_kernel='cubic',
                                           up_kernel='cubic', sf=args.scale[0], range=args.rgb_range)
            if args.rgb_range == 1:
                final_sr = np.clip(sr_np * 255, 0, args.rgb_range * 255)
            else:
                final_sr = np.clip(sr_np, 0, args.rgb_range)

            final_sr = final_sr.astype(np.uint8)
            final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.dir_out, os.path.split(img_lists[i])[-1]), final_sr)






if __name__ == '__main__':

    # args parameter setting
    args.pre_train = '/home/work/experiment/test/model/model_best.pt'
    args.dir_out = '/home/experiment/test/results'
    args.dir_data = '/home/data/AID-dataset/test/LR_x4/'


    checkpoint = utils.checkpoint(args)

    device = torch.device( "cuda:0" if torch.cuda.is_available() else"cpu")
    # device = torch.device("cpu")


    sr_model = model.Model(args, checkpoint).to(device)  # modify
    sr_model.eval()



    flops, params = get_model_complexity_info(sr_model, (3, 96, 96), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, ', params: ', params)


    deploy(args, sr_model)



