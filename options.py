import argparse
import os

class Options():

    def __init__(self):
        self.parser=argparse.ArgumentParser()

    def initialize(self,parser):

        # device
        parser.add_argument("--gpu_ids",type=str,default='0')
        parser.add_argument('--nums_works', type=int, default=1)
        parser.add_argument('--device', type=str, default='cuda')

        # dataset
        parser.add_argument('--batch', type=int, default=1)
        parser.add_argument('--img_size', type=int, default=(256,192))
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--joint_num', type=int, default=18)
        parser.add_argument('--human_sg_num', type=int, default=7, help='parser num')
        parser.add_argument('--sigma', type=int, default=6)

        # dir
        parser.add_argument('--data_dir', type=str, default='your dir')
        parser.add_argument('--cloth_dir',type=str,default='cloth')
        parser.add_argument('--cloth_mask_dir', type=str, default='cloth-mask')
        parser.add_argument('--human_dir', type=str, default='image')
        parser.add_argument('--human_mask_dir', type=str, default='image-parse\\sg')
        parser.add_argument('--densepose_dir', type=str, default='densepose')
        parser.add_argument('--openpose_dir', type=str, default='openpose')
        parser.add_argument('--vis_dir',type=str,default='vis')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
        parser.add_argument('--loss_dir', type=str, default='loss')
        parser.add_argument('--eval_dir', type=str, default='pred_sg')

        # model
        parser.add_argument('--submodel', type=str, default='',help='TPIM/PTM/RSIM')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--num_heads', type=int, default=16)
        parser.add_argument('--num_layers', type=int, default=16)
        parser.add_argument('--hidden_dim', type=int, default=4096)
        parser.add_argument('--g_lr',type=float,default=0.0001)
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--proj_dim', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--mlp-ratio', type=int, default=4)
        parser.add_argument('--drop', type=float, default=0.0)
        parser.add_argument('--atten_drop', type=float, default=0.0)
        parser.add_argument('--drop_path_rate', type=float, default=0.0)
        parser.add_argument('--tps_extract_layers', type=str, default=['relu1_1','relu2_1','relu3_1','relu4_1'])
        parser.add_argument('--tps_scale', type=int, default=[3,4,5,6])
        parser.add_argument('--tps_embed_dim', type=int, default=2 * 64)
        parser.add_argument('--tps_feat_size', type=int, default=[16, 12])
        parser.add_argument('--tps_level_num', type=int, default=4)
        parser.add_argument('--tps_tfm_depth', type=int, default=3, help="TPS_Transformer block num")
        parser.add_argument('--cloth_ngf', type=int, default=16)
        parser.add_argument('--composition_ngf', type=int, default=32)
        parser.add_argument('--vgg_layer', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        parser.add_argument('--pool_size', type=int, default=[3, 2])
        parser.add_argument('--reduce_chan', type=int, default=64)


        # hyper para
        parser.add_argument('--l1_loss_coeff', type=float, default=2.0)
        parser.add_argument('--ce_loss_coeff', type=float, default=0.2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--mlp-ratio', type=int, default=4)
        parser.add_argument('--drop', type=float, default=0.0)
        parser.add_argument('--atten_drop', type=float, default=0.0)
        parser.add_argument('--drop_path_rate', type=float, default=0.0)
        parser.add_argument('--warp_cloth_weight', type=float, default=[1.0,2.0,3.0])
        parser.add_argument('--coarse_loss_coeff', type=float, default=3.0)
        parser.add_argument('--fined_l1_loss_coeff', type=float, default=6.0)
        parser.add_argument('--fined_vgg_loss_coeff', type=float, default=0.2)
        parser.add_argument('--composition_loss_coeff', type=float, default=3.0)
        parser.add_argument('--grid_loss_coeff', type=float, default=0.3)
        parser.add_argument('--coarse_result_weight', type=float, default=0.3)
        parser.add_argument('--grouth_truth_weight', type=float, default=0.7)
        parser.add_argument('--rsim_l1_loss_coeff', type=float, default=200.0)
        parser.add_argument('--l2_loss_coeff', type=float, default=200.0)
        parser.add_argument('--content_loss_coeff', type=float, default=20.0)

        # train
        parser.add_argument('--mode', type=str, default='eval', help='train or eval')
        parser.add_argument('--inference', type=int, default=50,help='checkpoint loading')
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--vis_ite', type=int, default=400)
        parser.add_argument('--log_print_ite', type=int, default=50)
        parser.add_argument('--log_vis_ite', type=int, default=200)
        parser.add_argument('--save_epo', type=int, default=2)

        return parser

    def parse(self):

        parser=self.initialize(self.parser)
        opt,_=parser.parse_known_args()

        self.opt=opt

        return self.opt
