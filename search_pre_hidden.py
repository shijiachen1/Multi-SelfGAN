
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models_search
import models
import datasets
from function_controller_block import train_shared, train_distill, train_controller, get_topk_arch_hidden, train_KD, train_controller_block, get_topk_arch_with_hidden, get_top5_arch_with_hidden
from utils.utils import set_log_dir, save_checkpoint, create_logger, RunningStats
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from torchstat import stat
# from torchsummary import summary

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class GrowCtrler(object):
    def __init__(self, grow_step1, grow_step2):
        self.grow_step1 = grow_step1
        self.grow_step2 = grow_step2

    def cur_stage(self, search_iter):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        if search_iter < self.grow_step1:
            return 0
        elif self.grow_step1 <= search_iter < self.grow_step2:
            return 1
        else:
            return 2


def create_ctrler(args, cur_stage, weights_init):
    controller = eval('models_search. ' +args.controller_block + '.Controller')(args=args, cur_stage=cur_stage).cuda()
    # controller2 = eval('models_search.' + args.controller_block + '.Controller')(args=args, cur_stage=cur_stage).cuda()
    # controller3 = eval('models_search.' + args.controller_block + '.Controller')(args=args, cur_stage=cur_stage).cuda()
    controller.apply(weights_init)
    # controller1.apply(weights_init)
    # controller1.apply(weights_init)
    ctrl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, controller.parameters()),
                                      args.ctrl_lr, (args.beta1, args.beta2))
    # ctrl2_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, controller2.parameters()),
    #                                  args.ctrl_lr, (args.beta1, args.beta2))
    # ctrl3_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, controller3.parameters()),
    #                                  args.ctrl_lr, (args.beta1, args.beta2))
    return controller, ctrl_optimizer

# block-wise
# def create_shared_gan(args, weights_init):
#     # gen_net1 = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
#     # gen_net2 = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
#     # gen_net3 = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
#
#     gen_net = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
#     dis_net = eval('models_search.' + args.dis_model + '.Discriminator')(args=args).cuda()
#
#     # gen_net1.apply(weights_init)
#     # gen_net2.apply(weights_init)
#     # gen_net3.apply(weights_init)
#
#     gen_net.apply(weights_init)
#     dis_net.apply(weights_init)
#
#     cell1_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell1.parameters()),
#                                      args.g_lr, (args.beta1, args.beta2))
#     cell2_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell2.parameters()),
#                                       args.g_lr, (args.beta1, args.beta2))
#     cell3_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell3.parameters()),
#                                       args.g_lr, (args.beta1, args.beta2))
#
#     gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
#                                      args.g_lr, (args.beta1, args.beta2))
#     dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
#                                      args.d_lr, (args.beta1, args.beta2))
#     # return gen_net1,gen_net2,gen_net3, dis_net, gen_optimizer1, gen_optimizer2, gen_optimizer3, dis_optimizer
#     return gen_net, dis_net, gen_optimizer, dis_optimizer, cell1_opt, cell2_opt, cell3_opt

def create_shared_gan(args, weights_init):
    # gen_net1 = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    # gen_net2 = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
    # gen_net3 = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()

    gen_net = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
    dis_net = eval('models_search.' + args.dis_model + '.Discriminator')(args=args).cuda()

    # gen_net1.apply(weights_init)
    # gen_net2.apply(weights_init)
    # gen_net3.apply(weights_init)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # cell1_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell1.parameters()),
    #                                  args.g_lr, (args.beta1, args.beta2))
    # cell2_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell2.parameters()),
    #                                   args.g_lr, (args.beta1, args.beta2))
    # cell3_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.cell3.parameters()),
    #                                   args.g_lr, (args.beta1, args.beta2))

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    return gen_net, dis_net, gen_optimizer, dis_optimizer
    # return gen_net, gen_optimizer

def create_teacher_b_net(args):

    teacher_net = eval('models.' + args.teacher_gen_b + '.Generator')(args=args).cuda()
    teacher_net_dis = eval('models.' + args.teacher_gen_b + '.Discriminator')(args=args).cuda()
    # dis_net.load_state_dict(checkpoint['dis_state_dict'])
    return teacher_net, teacher_net_dis

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gpu = [0]
    # gen_net1,gen_net2,gen_net3, dis_net, gen_optimizer1, gen_optimizer2, gen_optimizer3, dis_optimizer = create_shared_gan(args, weights_init)
    gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)

    # checkpoint_file = args.teacher_gen_b_path

    checkpoint_file_256 = args.teacher_gen_b_256_path
    checkpoint_file = args.teacher_gen_b_path

    assert os.path.exists(checkpoint_file)
    assert os.path.exists(checkpoint_file_256)

    checkpoint_256 = torch.load(checkpoint_file_256)
    checkpoint = torch.load(checkpoint_file)
    # torch.save(checkpoint, checkpoint_file, _use_new_zipfile_serialization=False)

    teacher_net_b, teacher_net_dis = create_teacher_b_net(args)

    teacher_net_b.load_state_dict(checkpoint['gen_state_dict'])
    teacher_net_dis.load_state_dict(checkpoint_256['dis_state_dict'])

    teacher_net_b.eval()
    teacher_net_dis.eval()
    # summary(teacher_net_b)

    # initial
    start_search_iter = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        # set controller && its optimizer
        # cur_stage = checkpoint['cur_stage']
        controller1, ctrl_optimizer1 = create_ctrler(args, 0, weights_init)
        controller2, ctrl_optimizer2 = create_ctrler(args, 1, weights_init)
        controller3, ctrl_optimizer3 = create_ctrler(args, 2, weights_init)

        start_search_iter = checkpoint['search_iter']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        controller1.load_state_dict(checkpoint['ctrl_state1_dict'])
        controller2.load_state_dict(checkpoint['ctrl_state2_dict'])
        controller3.load_state_dict(checkpoint['ctrl_state3_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        ctrl_optimizer1.load_state_dict(checkpoint['ctrl_optimizer1'])
        ctrl_optimizer2.load_state_dict(checkpoint['ctrl_optimizer2'])
        ctrl_optimizer3.load_state_dict(checkpoint['ctrl_optimizer3'])
        prev_archs = checkpoint['prev_archs']
        if prev_archs:
            prev_hiddens = checkpoint['prev_hiddens']
            prev_hxs, prev_cxs = prev_hiddens
        else:
            prev_hiddens = None

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (search iteration {start_search_iter})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        prev_archs = None
        prev_hiddens = None

        # set controller && its optimizer
        controller1, ctrl_optimizer1 = create_ctrler(args, 0, weights_init)
        controller2, ctrl_optimizer2 = create_ctrler(args, 1, weights_init)
        controller3, ctrl_optimizer3 = create_ctrler(args, 2, weights_init)

    # set up data_loader
    dataset = datasets.ImageDataset(args, 2** (2 + 3))
    train_loader = dataset.train
    eval_loader = dataset.valid

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'controller_steps': start_search_iter * args.ctrl_step
    }

    g_loss_history = RunningStats(args.dynamic_reset_window)
    d_loss_history = RunningStats(args.dynamic_reset_window)

    # train loop
    for search_iter in tqdm(range(int(start_search_iter), int(args.max_search_iter)), desc='search progress'):
        logger.info(f"<start search iteration {search_iter}>")

        # if search_iter != 0 and search_iter % 15 == 0:
        #     inter_top_archs, _ = get_top5_arch_with_hidden(args, controller1, controller2, controller3, gen_net, prev_archs=prev_archs, prev_hiddens=prev_hiddens)
        #     logger.info(f"discovered archs: {inter_top_archs}")

        if search_iter == args.grow_step1 or search_iter == args.grow_step2 or search_iter == args.grow_step3:
            prev_archs, prev_hiddens = get_top5_arch_with_hidden(args, controller1, controller2, controller3, gen_net, prev_archs=prev_archs, prev_hiddens=prev_hiddens)
            logger.info(f"discovered archs: {prev_archs}")

        # dynamic_reset = train_KD(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, controller1,
        #                          controller2, controller3, train_loader, g_loss_history, d_loss_history)

        dynamic_reset = train_shared(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, controller1,
                                 controller2, controller3, train_loader, g_loss_history, d_loss_history, prev_archs=prev_archs, prev_hiddens=prev_hiddens)

        train_controller_block(args, controller1, controller2, controller3, ctrl_optimizer1, ctrl_optimizer2, ctrl_optimizer3,
                               teacher_net_b, teacher_net_dis, gen_net, dis_net, eval_loader, prev_archs=prev_archs, prev_hiddens=prev_hiddens)

        if dynamic_reset:
            logger.info('re-initialize share GAN')
            del gen_net, dis_net, gen_optimizer, dis_optimizer
            gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)

        save_checkpoint({
            # 'cur_stage': cur_stage,
            'search_iter': search_iter + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'controller': args.controller_block,

            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'ctrl_state1_dict': controller1.state_dict(),
            'ctrl_state2_dict': controller2.state_dict(),
            'ctrl_state3_dict': controller3.state_dict(),

            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'ctrl_optimizer1': ctrl_optimizer1.state_dict(),
            'ctrl_optimizer2': ctrl_optimizer2.state_dict(),
            'ctrl_optimizer3': ctrl_optimizer3.state_dict(),

            'prev_archs': prev_archs,
            'prev_hiddens': prev_hiddens,
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'])

    final_archs,_ = get_top5_arch_with_hidden(args, controller1, controller2, controller3, gen_net, prev_archs=prev_archs, prev_hiddens=prev_hiddens)
    logger.info(f"discovered archs: {final_archs}")


if __name__ == '__main__':
    main()
