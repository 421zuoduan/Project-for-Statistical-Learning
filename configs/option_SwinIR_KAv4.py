import argparse
import os
from UDL.Basis.python_sub_class import TaskDispatcher


class parser_args(TaskDispatcher, name='SwinIR_KAv4'):

    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))

        root_dir = script_path.split('AutoDL')[0]

        # model_path = ''
        model_path = ''

        parser = argparse.ArgumentParser(
            description='PyTorch ImageNet Training')
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{cfg.task}', help='path to save model')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='SwinIR_KAv4')
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true', help= "If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'), help="Type of positional embedding to use on top of the image features")

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

        ## Train
        parser.add_argument('--patch_size', type=int, default=48, help='image2patch, set to model and dataset')
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('-samples_per_gpu', default=6, type=int, metavar='N',help='mini-batch size (default: 256)')        # 8 / 4
        parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
        parser.add_argument('--device', default='cuda', help='device to use for training / testing')
        parser.add_argument('--epochs', default=1001, type=int)
        parser.add_argument('--workers_per_gpu', default=4, type=int)
        parser.add_argument('--resume_from', default=model_path, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--clip_max_norm', default=0, type=float, help='gradient clipping max norm')
        parser.add_argument('--dataset',default={'train': "Rain100L", 'val': "Rain100L"}, type=str, 
                            choices=[None, 'Rain200L', 'Rain100L', 'Rain200H', 'Rain100H', 'test12', 'real', 'DID', 'SPA'],help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool, help="performing evalution for patch2entire")
        parser.add_argument('--crop_batch_size', type=int, default=64, help='input batch size for training')
        parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')

        # SRData
        parser.add_argument('--model', default='SwinIR_KAv4', help='model name')
        parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
        parser.add_argument('--data_train', type=str, default='RainTrainL', help='train dataset name')  # DIV2K 
        parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')
        parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
        parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')

        # Model specifications
        parser.add_argument('--task_head', type=str, default='derain',
                            help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')


        args = parser.parse_args()
        args.experimental_desc = "Test"
        args.adjust_size_mode = "patch"
        args.workflow = [('train', 1)]

        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)
