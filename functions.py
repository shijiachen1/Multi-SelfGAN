
# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score

logger = logging.getLogger(__name__)


def train_shared(args, teacher_net_b, gen_net, dis_net, gen_optimizer, cell1_opt, cell2_opt, cell3_opt, g_loss_history, d_loss_history,
                 controller, dis_optimizer, train_loader):
    dynamic_reset = False
    logger.info('=> train shared GAN...')
    step = 0
    gen_step = 0

    mse_loss_fn = torch.nn.L1Loss().cuda()
    # train mode
    # gen_net1.train()
    # gen_net2.train()
    # gen_net3.train()
    gen_net.train()
    dis_net.train()
    teacher_net_b.eval()

    # eval mode
    controller.eval()
    # controller2.eval()
    # controller3.eval()
    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):

            # sample an arch
            arch = controller.sample(1)[0][0]
            # print(arch)
            # arch2 = controller2.sample(1)[0][0]
            # arch3 = controller3.sample(1)[0][0]
            # arch = torch.cat([arch1,arch2,arch3])
            # print(arch1)
            # print(arch2)
            # print(arch3)
            gen_net.set_arch(arch)
            # gen_net1.set_arch(arch1)
            # gen_net2.set_arch(arch2)
            # gen_net3.set_arch(arch3)

            # dis_net.cur_stage = controller.cur_stage
            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            _,_,fake_imgs = gen_net(z)
            assert fake_imgs.size() == real_imgs.size(), print(f'fake image size is {fake_imgs.size()}, '
                                                               f'while real image size is {real_imgs.size()}')

            fake_validity = dis_net(fake_imgs.detach())

            # cal loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:

                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

                h1, h2, h3 = teacher_net_b(gen_z)
                stu_h1, stu_h2, stu_h3, stu_imgs_3 = gen_net([gen_z, h1, h2], tea=True)

                # fake_validity_stu = dis_net(stu_imgs_3)
                # fake_validity_tea = dis_net(tea_imgs_3)
                cell1_loss = mse_loss_fn(h1[0].detach(), stu_h1)
                cell1_opt.zero_grad()
                cell1_loss.backward()
                cell1_opt.step()

                cell2_loss = mse_loss_fn(h2[0].detach(), stu_h2)
                cell2_opt.zero_grad()
                cell2_loss.backward()
                cell2_opt.step()

                cell3_loss = mse_loss_fn(h3.detach(),stu_h3)
                cell3_opt.zero_grad()
                cell3_loss.backward()
                cell3_opt.step()

                fake_validity = torch.mean(dis_net(stu_imgs_3))
                g_loss = -fake_validity
                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()

                # loss = mse_loss_fn(stu_imgs_3, tea_imgs_3)
                # print('l1_loss:{}'.format(loss))
                # fake_validity = -torch.mean(dis_net(stu_imgs_3))
                # loss = loss + fake_validity

                # add to window
                g_loss_history.push(g_loss.item())
                gen_step += 1

            # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(),
                     g_loss.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        _, _, fake_imgs = gen_net(z)
        # print(fake_imgs)
        assert fake_imgs.size() == real_imgs.size(), print(f'fake image size is {fake_imgs.size()}, '
                                                           f'while real image size is {real_imgs.size()}')

        fake_validity = dis_net(fake_imgs.detach())

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            _, _, gen_imgs = gen_net(gen_z)

            # train_entirely
            fake_validity = torch.mean(dis_net(gen_imgs))
            g_loss = -fake_validity
            g_loss.backward()

            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_distill(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              schedulers):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    teacher_net_b = teacher_net_b.eval()

    mse_loss_fn = torch.nn.MSELoss().cuda()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        _, _, fake_imgs = gen_net(z)
        # print(fake_imgs)
        assert fake_imgs.size() == real_imgs.size(), print(f'fake image size is {fake_imgs.size()}, '
                                                           f'while real image size is {real_imgs.size()}')

        fake_validity = dis_net(fake_imgs.detach())

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

            # stu_imgs_1 = gen_net1(gen_z)
            # stu_imgs_2 = gen_net2(stu_imgs_1)
            # stu_imgs_3 = gen_net3(stu_imgs_2)
            stu_imgs_1, stu_imgs_2, stu_imgs_3 = gen_net(gen_z)
            tea_imgs_1, tea_imgs_2, tea_imgs_3 = teacher_net_b(gen_z)
            # tea_imgs_2 = teacher_net_b(tea_imgs_1)
            # tea_imgs_3 = teacher_net_b(tea_imgs_2)

            fake_gan = -torch.mean(dis_net(stu_imgs_3))
            # fake_validity.backward()
            # gen_optimizer.step()
            #
            # mse_loss_1 = mse_loss_fn(stu_imgs_1, tea_imgs_1)
            #
            # mse_loss_2 = mse_loss_fn(stu_imgs_2, tea_imgs_2)

            mse_loss_3 = mse_loss_fn(stu_imgs_3, tea_imgs_3)

            g_loss = fake_gan + mse_loss_3

            g_loss.backward()
            gen_optimizer.step()

            g_loss = fake_validity

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), fake_gan.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def train_controller(args, controller, ctrl_optimizer, teacher_net_b, gen_net):
    logger.info("=> train controller...")
    # writer = writer_dict['writer']
    baseline = None
    # baseline2 = None
    # baseline3 = None

    # train mode
    controller.train()
    # controller2.train()
    # controller3.train()

    # eval mode
    # gen_net1.eval()
    # gen_net2.eval()
    # gen_net3.eval()
    gen_net.eval()
    eval_iter = args.rl_num_eval_img // args.eval_batch_size

    # cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        for i in range(eval_iter):
            # for iter_idx, (imgs, _) in enumerate(eval_loader):
            # controller_step = writer_dict['controller_steps']
            archs, entropies ,selected_log_probs = controller.sample(args.ctrl_sample_batch)

            modified_batch_rewards = entropies
            # archs2, entropies2,selected_log_probs2 = controller2.sample(args.ctrl_sample_batch)
            # archs3, entropies3,selected_log_probs3 = controller3.sample(args.ctrl_sample_batch)
            cur_batch_rewards = []
            for arch in archs:
                # arch = torch.cat([arch1,arch2,arch3])
                gen_net.set_arch(arch)

                # ctrl1_batch_rewards = []
                # ctrl2_batch_rewards = []
                # ctrl3_batch_rewards = []
                # logger.info(f'arch: {arch}')
                # gen_net1.set_arch(arch1)
                # gen_net2.set_arch(arch2)
                # gen_net3.set_arch(arch3)
                # is_score = get_is(args, gen_net, args.rl_num_eval_img)
                action_value = get_reward(args, gen_net, teacher_net_b)
                # logger.info(f"-->Cosine Similarity : {action_value}")
                cur_batch_rewards.append(action_value)
                # ctrl1_batch_rewards.append(action_value[0])
                # ctrl2_batch_rewards.append(action_value[1])
                # ctrl3_batch_rewards.append(action_value[2])
            cur_batch_rewards = torch.cat(cur_batch_rewards).view(args.ctrl_sample_batch, args.total_cells)
            cur_batch_rewards = cur_batch_rewards.unsqueeze(-1)

            # ctrl1_batch_rewards = ctrl1_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies1  # bs * 1
            # ctrl2_batch_rewards = ctrl2_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies2
            # ctrl3_batch_rewards = ctrl3_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies3
            modified_batch_rewards[:, 0:4] = args.entropy_coeff * entropies[:, 0:4] + 0.2*cur_batch_rewards[:, 0] + 0.8*cur_batch_rewards[:, 2]
            modified_batch_rewards[:, 4:9] = args.entropy_coeff * entropies[:, 4:9] + 0.4*cur_batch_rewards[:, 1] + 0.6*cur_batch_rewards[:, 2]
            modified_batch_rewards[:, 9:] = args.entropy_coeff * entropies[:, 9:] + cur_batch_rewards[:, 2]

            cur_batch_rewards = modified_batch_rewards
            if baseline is None:
                baseline = cur_batch_rewards
            else:
                baseline = args.baseline_decay * baseline.detach() + (1 - args.baseline_decay) * cur_batch_rewards
            adv = cur_batch_rewards - baseline

            # policy loss
            loss = -selected_log_probs * adv  # 每个reward加上关于所有可能操作的entropy * action的概率
            loss = loss.sum()

            ctrl_optimizer.zero_grad()
            loss.backward()
            ctrl_optimizer.step()
            # if baseline1 and baseline2 and baseline3:
            #     baseline1 = args.baseline_decay * baseline1.detach() + (1 - args.baseline_decay) * ctrl1_batch_rewards
            #     baseline2 = args.baseline_decay * baseline2.detach() + (1 - args.baseline_decay) * ctrl2_batch_rewards
            #     baseline3 = args.baseline_decay * baseline3.detach() + (1 - args.baseline_decay) * ctrl3_batch_rewards
            # else:
            #     baseline1 = ctrl1_batch_rewards
            #     baseline2 = ctrl2_batch_rewards
            #     baseline3 = ctrl3_batch_rewards
            #
            # adv1 = ctrl1_batch_rewards - baseline1
            # adv2 = ctrl1_batch_rewards - baseline2
            # adv3 = ctrl1_batch_rewards - baseline3
            #
            # # policy loss
            # loss1 = -selected_log_probs1 * adv1
            # loss1 = loss1.sum()
            # loss2 = -selected_log_probs2 * adv2
            # loss2 = loss2.sum()
            # loss3 = -selected_log_probs3 * adv3
            # loss3 = loss3.sum()
            #
            # # update controller
            # ctrl_optimizer1.zero_grad()
            # loss1.backward()
            # ctrl_optimizer1.step()
            #
            # ctrl_optimizer2.zero_grad()
            # loss2.backward()
            # ctrl_optimizer2.step()
            #
            # ctrl_optimizer3.zero_grad()
            # loss3.backward()
            # ctrl_optimizer3.step()
            # verbose
            if step and i % 10 == 0:
                logger.info(f'arch: {archs}')
                logger.info(f'Cosine_Similarity: {action_value}')
                tqdm.write(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                    (step, args.ctrl_step, i % eval_iter, eval_iter, loss.item()))

def transform(imgs_list, batch_size):
    imgs = []
    for img in imgs_list:
        imgs.append(img.view(batch_size ,-1))
    return imgs

def get_IS(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        _, _, gen_imgs = gen_net(z)
        gen_imgs = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                              torch.uint8).numpy()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('calculate Inception score...')
    mean, std = get_inception_score(img_list)

    return mean

def get_reward(args, gen_net, teacher_net_b):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()
    gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    # for iter_idx, (imgs, _) in enumerate(eval_loader):
    # eval_iter = args.rl_num_eval_img // args.eval_batch_size
    # cells_reward = []
    # cell2_reward = []
    # cell3_reward = []
    # for _ in range(eval_iter):
    cell_reward = []
    stu_imgs1, stu_imgs2, stu_imgs3 = transform(gen_net(gen_z) ,args.eval_batch_size)
    tea_imgs1, tea_imgs2, tea_imgs3 = transform(teacher_net_b(gen_z) ,args.eval_batch_size)
    cell_reward.append(torch.cosine_similarity(stu_imgs1, tea_imgs1 ,dim=1).mean())
    cell_reward.append(torch.cosine_similarity(stu_imgs2, tea_imgs2, dim=1).mean())
    cell_reward.append(torch.cosine_similarity(stu_imgs3, tea_imgs3, dim=1).mean())
    # cells_reward.append(cell_reward)

    cell_reward = torch.tensor(cell_reward, requires_grad=False).cuda()
    # cell2_reward = torch.tensor(cell2_reward, requires_grad=False).cuda()
    # cell3_reward = torch.tensor(cell3_reward, requires_grad=False).cuda()

    return cell_reward
    #     z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    #
    #     # Generate a batch of images
    #     # gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
    #     #                                                                                         torch.uint8).numpy()
    #     gen_imgs = gen_net(z)
    #     gen_performance = dis_net(gen_imgs)
    #     performance_list.append(torch.sum(gen_performance))
    #
    # # get inception score
    # logger.info('calculate Inception score...')
    # #mean, std = get_inception_score(img_list)
    # #gen_performance = dis_net(torch.tensor(img_list).cuda())
    #
    # #return mean
    # return torch.mean(torch.Tensor(performance_list))


def validate(args, fixed_z, fid_stat, gen_net, writer_dict, clean_dir=True):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net.eval()

    # generate images
    _, _, sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        _, _, gen_imgs = gen_net(z)
        gen_imgs = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                              torch.uint8).numpy()
        # pyplot(gen_imgs)
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)
    print(f"Inception score: {mean}")
    # logger.info('=> calculate generator performance')
    # gen_performance = dis_net(img_list)
    # gen_performance = torch.sum(gen_performance)
    # print(f"Inception score: {gen_performance}")

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    print(f"FID score: {fid_score}")

    if clean_dir:
        os.system('rm -r {}'.format(fid_buffer_dir))
    else:
        logger.info(f'=> sampled images are saved to {fid_buffer_dir}')

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)
    # writer.add_scalar('Gen_performance', gen_performance)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score

def get_topk_arch_hidden(args, controller, gen_net):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()

    archs = controller.sample(args.num_candidate)[0]
    # hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx])
        is_score = get_IS(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]
    topk_archs = []
    # topk_hxs = []
    # topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        # topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        # topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten





