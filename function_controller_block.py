
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


def train_KD(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, controller1, controller2, controller3,
             train_loader, g_loss_history, d_loss_history):
    dynamic_reset = False
    logger.info('=> train shared GAN...')
    step = 0
    gen_step = 0

    mse_loss_fn = torch.nn.MSELoss().cuda()
    L1_loss_fn = torch.nn.L1Loss().cuda()
    # train mode
    gen_net.train()
    teacher_net_b.eval()

    # eval mode
    controller1.eval()
    controller2.eval()
    controller3.eval()

    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):

            # sample an arch
            arch1 = controller1.sample(1)[0][0]
            arch2 = controller2.sample(1)[0][0]
            arch3 = controller3.sample(1)[0][0]
            arch = torch.cat([arch1,arch2,arch3], dim=-1)
            # print(arch)
            gen_net.set_arch(arch)

            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            _, _, stu_fake_imgs = gen_net(z)
            _, _, tea_fake_imgs = teacher_net_b(z)
            assert stu_fake_imgs.size() == real_imgs.size() == tea_fake_imgs.size(), print(f'stu_fake image size is {stu_fake_imgs.size()}, '
                                                               f'tea_fake image size is {tea_fake_imgs.size()}, '
                                                               f'while real image size is {real_imgs.size()}')

            stu_fake_validity = dis_net(stu_fake_imgs.detach())
            tea_fake_validity = dis_net(tea_fake_imgs.detach())

            # cal loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1.0 + stu_fake_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1.0 + tea_fake_validity))

            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:
                gen_optimizer.zero_grad()
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

                _, _, tea_img3 = teacher_net_b(gen_z)
                _, _, stu_img3 = gen_net(gen_z)

                KD_loss = L1_loss_fn(stu_img3, tea_img3)
                # Per_loss = mse_loss_fn(dis_net(stu_img3), dis_net(tea_img3))
                fake_gan_loss = -torch.mean(dis_net(stu_img3))

                g_loss = KD_loss + fake_gan_loss

                g_loss.backward()
                gen_optimizer.step()
                g_loss_history.push(fake_gan_loss.item())

                gen_step += 1

            # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [mixed_distill D loss: %f] [mixed_distill G loss: %f] [fake_gan loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), fake_gan_loss.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    # print('yes')
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset

def train_shared(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, controller1,
                controller2, controller3, train_loader, g_loss_history, d_loss_history, prev_archs, prev_hiddens):
    dynamic_reset = False
    logger.info('=> train shared GAN...')
    step = 0
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()

    # eval mode
    teacher_net_b.eval()
    controller1.eval()
    controller2.eval()
    controller3.eval()

    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):
            with torch.no_grad():
                if prev_hiddens:
                    prev_hxs1 = []
                    prev_hxs2 = []
                    prev_hxs3 = []

                    prev_cxs1 = []
                    prev_cxs2 = []
                    prev_cxs3 = []
                    hxs, cxs = prev_hiddens
                    # print("hxs:{}".format(hxs))

                    for (hx, cx) in zip(hxs, cxs):
                        prev_hxs1.append(hx[0:args.hid_size])
                        prev_hxs2.append(hx[args.hid_size:args.hid_size * 2])
                        prev_hxs3.append(hx[args.hid_size * 2:args.hid_size * 3])

                        prev_cxs1.append(cx[0:args.hid_size])
                        prev_cxs2.append(cx[args.hid_size:args.hid_size * 2])
                        prev_cxs3.append(cx[args.hid_size * 2:args.hid_size * 3])

                    # prev_hxs1 = hxs[:, 0:args.hid_size]
                    # prev_hxs2 = hxs[:, args.hid_size:args.hid_size * 2]
                    # prev_hxs3 = hxs[:, args.hid_size * 2:args.hid_size * 3]
                    #
                    # prev_cxs1 = cxs[:, 0:args.hid_size]
                    # prev_cxs2 = cxs[:, args.hid_size:args.hid_size * 2]
                    # prev_cxs3 = cxs[:, args.hid_size * 2:args.hid_size * 3]

                    prev_hiddens1 = (prev_hxs1, prev_cxs1)
                    prev_hiddens2 = (prev_hxs2, prev_cxs2)
                    prev_hiddens3 = (prev_hxs3, prev_cxs3)

                    # archs1, _,_, hiddens1 = controller1.sample(args.num_candidate, with_hidden=True)
                    # archs2, _,_, hiddens2 = controller2.sample(args.num_candidate, prev_hiddens=prev_hiddens1, with_hidden=True)
                    # archs3, _,_, hiddens3 = controller3.sample(args.num_candidate, prev_hiddens=prev_hiddens2, with_hidden=True)

                    archs1, _, _, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True,
                                                                cur_top_hiddens=prev_hiddens1)
                    archs2, _, _, hiddens2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=hiddens1,
                                                                with_hidden=True, cur_top_hiddens=prev_hiddens2)
                    archs3, _, _, hiddens3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=hiddens2,
                                                                with_hidden=True, cur_top_hiddens=prev_hiddens3)

                else:
                    archs1, _, _, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
                    archs2, _, _, hiddens2 = controller2.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens1)
                    archs3, _, _, hiddens3 = controller3.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens2)

            # arch1,_,_, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
            # arch2,_,_, hiddens2 = controller2.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens1)
            # arch3,_,_, hiddens3 = controller3.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens2)

            arch = torch.cat([archs1, archs2, archs3], dim=-1)

            gen_net.set_arch(arch[0])

            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            _, _, stu_fake_imgs = gen_net(z)
            # _, _, tea_fake_imgs = teacher_net_b(z)

            # assert stu_fake_imgs.size() == real_imgs.size() == tea_fake_imgs.size(), print(f'stu_fake image size is {stu_fake_imgs.size()}, '
            #                                                    f'tea_fake image size is {tea_fake_imgs.size()}, '
            #                                                    f'while real image size is {real_imgs.size()}')

            assert stu_fake_imgs.size() == real_imgs.size(), print(f'stu_fake image size is {stu_fake_imgs.size()},'
                                                               f'while real image size is {real_imgs.size()}')

            stu_fake_validity = dis_net(stu_fake_imgs.detach())
            # tea_fake_validity = dis_net(tea_fake_imgs.detach())

            # cal loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1.0 + stu_fake_validity))

            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:
                gen_optimizer.zero_grad()
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

                # _, _, tea_img3 = teacher_net_b(gen_z)
                _, _, stu_img3 = gen_net(gen_z)

                fake_gan_loss = -torch.mean(dis_net(stu_img3))

                g_loss = fake_gan_loss

                g_loss.backward()
                gen_optimizer.step()
                g_loss_history.push(fake_gan_loss.item())

                gen_step += 1

            # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [fake_gan loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), fake_gan_loss.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    # print('yes')
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset

def train_shared_combine(args, teacher_net_b, gen_net, dis_net, gen_optimizer, dis_optimizer, controller, train_loader):
    logger.info('=> train shared GAN...')
    step = 0
    gen_step = 0

    mse_loss_fn = torch.nn.L1Loss().cuda()
    # train mode
    gen_net.train()
    teacher_net_b.eval()

    # eval mode
    controller.eval()

    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):

            # sample an arch
            arch = controller.sample(1)[0][0]
            gen_net.set_arch(arch)

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

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:

                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

                _, _, tea_img3 = teacher_net_b(gen_z)
                # print(h3)
                _, _, stu_img3 = gen_net(gen_z)

                fake_gan = -torch.mean(dis_net(stu_img3))

                g_loss = mse_loss_fn(stu_img3, tea_img3) + fake_gan

                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()

                gen_step += 1

                # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D_loss: %f] [G_loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), fake_gan.item()))
            step += 1

def train(args, gen_net: nn.Module, dis_net: nn.Module, teacher_net_b, teacher_net_dis, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
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
              lr_schedulers):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
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
        _, _, stu_fake_imgs = gen_net(z)
        _, _, tea_fake_imgs = teacher_net_b(z)
        assert stu_fake_imgs.size() == real_imgs.size() == tea_fake_imgs.size(), print(
            f'stu_fake image size is {stu_fake_imgs.size()}, '
            f'tea_fake image size is {tea_fake_imgs.size()}, '
            f'while real image size is {real_imgs.size()}')

        stu_fake_validity = dis_net(stu_fake_imgs.detach())
        tea_fake_validity = dis_net(tea_fake_imgs.detach())

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + stu_fake_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 - tea_fake_validity))

        d_loss.backward()
        dis_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

            _, _, tea_img3 = teacher_net_b(gen_z)
            _, _, stu_img3 = gen_net(gen_z)

            # KD_loss = L1_loss_fn(stu_img3, tea_img3)
            Per_loss = mse_loss_fn(dis_net(stu_img3), dis_net(tea_img3))
            fake_gan_loss = -torch.mean(dis_net(stu_img3))

            g_loss = 0.1 * Per_loss + fake_gan_loss

            g_loss.backward()
            gen_optimizer.step()


            # adjust learning rate
            if lr_schedulers:
                gen_scheduler, dis_scheduler = lr_schedulers
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
            logger.info(
                "[Batch %d/%d] [mixed_distill D loss: %f] [mixed_distill G loss: %f] [fake_gan loss: %f]" %
                (
                iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(),
                fake_gan_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_controller(args, controller, ctrl_optimizer, teacher_net_b, gen_net, dis_net):
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
                action_value = get_reward_withEM(args, gen_net, dis_net, teacher_net_b)
                cur_batch_rewards.append(action_value)

            cur_batch_rewards = torch.cat(cur_batch_rewards).view(args.ctrl_sample_batch, args.total_cells)
            cur_batch_rewards = cur_batch_rewards.unsqueeze(-1)

            # modified_batch_rewards[:, 0:4] = args.entropy_coeff * entropies[:, 0:4] + 0.2*cur_batch_rewards[:, 0] + 0.8 * cur_batch_rewards[:, 2]
            # modified_batch_rewards[:, 4:9] = args.entropy_coeff * entropies[:, 4:9] + 0.4*cur_batch_rewards[:, 1] + 0.6 * cur_batch_rewards[:, 2]
            # modified_batch_rewards[:, 9:] = args.entropy_coeff * entropies[:, 9:] + cur_batch_rewards[:, 2]

            modified_batch_rewards[:, 0:4] = args.entropy_coeff * entropies[:, 0:4] + cur_batch_rewards[:, 2]
            modified_batch_rewards[:, 4:9] = args.entropy_coeff * entropies[:, 4:9] + cur_batch_rewards[:, 2]
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
                logger.info(f'Cosine_Similarity: {action_value[2]}')
                tqdm.write(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                    (step, args.ctrl_step, i % eval_iter, eval_iter, loss.item()))

def train_controller_block(args, controller1, controller2, controller3, ctrl_optimizer1, ctrl_optimizer2, ctrl_optimizer3,
                               teacher_net_b, teacher_net_dis, gen_net, dis_net, eval_loader, prev_archs, prev_hiddens):
    logger.info("=> train controller...")
    # writer = writer_dict['writer']
    baseline1 = None
    baseline2 = None
    baseline3 = None

    # train mode
    controller1.train()
    controller2.train()
    controller3.train()

    # eval mode
    gen_net.eval()

    # cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        # for i in range(eval_iter):
        ctrl_optimizer1.zero_grad()
        ctrl_optimizer2.zero_grad()
        ctrl_optimizer3.zero_grad()

        if prev_hiddens:
            prev_hxs1 = []
            prev_hxs2 = []
            prev_hxs3 = []

            prev_cxs1 = []
            prev_cxs2 = []
            prev_cxs3 = []
            hxs, cxs = prev_hiddens
            # print("hxs:{}".format(hxs))
            with torch.no_grad():
                for (hx, cx) in zip(hxs, cxs):
                    prev_hxs1.append(hx[0:args.hid_size])
                    prev_hxs2.append(hx[args.hid_size:args.hid_size * 2])
                    prev_hxs3.append(hx[args.hid_size * 2:args.hid_size * 3])

                    prev_cxs1.append(cx[0:args.hid_size])
                    prev_cxs2.append(cx[args.hid_size:args.hid_size * 2])
                    prev_cxs3.append(cx[args.hid_size * 2:args.hid_size * 3])

                prev_hiddens1 = (prev_hxs1, prev_cxs1)
                prev_hiddens2 = (prev_hxs2, prev_cxs2)
                prev_hiddens3 = (prev_hxs3, prev_cxs3)

            # archs1, entropies1, selected_log_probs1, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
            # archs2, entropies2, selected_log_probs2, hiddens2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens1, with_hidden=True)
            # archs3, entropies3, selected_log_probs3, hiddens3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens2, with_hidden=True)

            archs1, entropies1, selected_log_probs1, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True,
                                                        cur_top_hiddens=prev_hiddens1)
            archs2, entropies2, selected_log_probs2, hiddens2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=hiddens1,
                                                        with_hidden=True, cur_top_hiddens=prev_hiddens2)
            archs3, entropies3, selected_log_probs3, hiddens3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=hiddens2,
                                                        with_hidden=True, cur_top_hiddens=prev_hiddens3)

        else:
            archs1, entropies1, selected_log_probs1, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
            archs2, entropies2, selected_log_probs2, hiddens2 = controller2.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens1)
            archs3, entropies3, selected_log_probs3, hiddens3 = controller3.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens2)

        # archs1, entropies1, selected_log_probs1 = controller1.sample(args.ctrl_sample_batch, cur_top_hiddens=prev_hiddens1)
        # archs2, entropies2, selected_log_probs2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens1, cur_top_hiddens=prev_hiddens2)
        # archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens2, cur_top_hiddens=prev_hiddens3)

        # archs1, entropies1, selected_log_probs1 = controller1.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens1)
        # archs2, entropies2, selected_log_probs2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens2)
        # archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens3)

        # archs1, entropies1, selected_log_probs1, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
        # archs2, entropies2, selected_log_probs2, hiddens2 = controller2.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens1)
        # archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=hiddens2)

        # cur_batch_rewards = []
        cur_batch_rewards1 = []
        cur_batch_rewards2 = []
        cur_batch_rewards3 = []
        archs = torch.cat([archs1,archs2,archs3],dim=-1)
        for (i, arch) in enumerate(archs):
            gen_net.set_arch(arch)
            # action_value, EM3, cos_sims = get_reward_block(args, gen_net, teacher_net_b, teacher_net_dis)
            # EMs, staged_EMs_stu, entire_EMs_stu, entire_dis_staged_EM_stu, entire_fake_validity_stu, staged_fake_validity_stu, action_value = get_reward_dis(args, gen_net, teacher_net_b, teacher_net_dis, eval_loader, train_epoch, arch, prev_archs)
            action_value = get_reward_rollout(args, gen_net, dis_net, hiddens1, hiddens2, controller2, controller3, teacher_net_b, teacher_net_dis, eval_loader, archs1[i], archs2[i], archs3[i], prev_archs)

            cur_batch_rewards1.append(action_value[0])
            cur_batch_rewards2.append(action_value[1])
            cur_batch_rewards3.append(action_value[2])

        cur_batch_rewards1 = torch.tensor(cur_batch_rewards1, requires_grad=False).cuda()
        cur_batch_rewards1 = cur_batch_rewards1.unsqueeze(-1) + args.entropy_coeff * entropies1

        cur_batch_rewards2 = torch.tensor(cur_batch_rewards2, requires_grad=False).cuda()
        cur_batch_rewards2 = cur_batch_rewards2.unsqueeze(-1) + args.entropy_coeff * entropies2

        cur_batch_rewards3 = torch.tensor(cur_batch_rewards3, requires_grad=False).cuda()
        cur_batch_rewards3 = cur_batch_rewards3.unsqueeze(-1) + args.entropy_coeff * entropies3

        if baseline1 is None and baseline2 is None and baseline3 is None:
            baseline1 = cur_batch_rewards1
            baseline2 = cur_batch_rewards2
            baseline3 = cur_batch_rewards3
        else:
            baseline1 = args.baseline_decay * baseline1.detach() + (1 - args.baseline_decay) * cur_batch_rewards1
            baseline2 = args.baseline_decay * baseline2.detach() + (1 - args.baseline_decay) * cur_batch_rewards2
            baseline3 = args.baseline_decay * baseline3.detach() + (1 - args.baseline_decay) * cur_batch_rewards3

        adv1 = cur_batch_rewards1 - baseline1
        adv2 = cur_batch_rewards2 - baseline2
        adv3 = cur_batch_rewards3 - baseline3

        # policy loss
        loss1 = -selected_log_probs1 * adv1  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss1 = loss1.sum()

        loss1.backward(retain_graph=True)
        ctrl_optimizer1.step()

        loss2 = -selected_log_probs2 * adv2  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss2 = loss2.sum()

        loss2.backward(retain_graph=True)
        ctrl_optimizer2.step()

        loss3 = -selected_log_probs3 * adv3  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss3 = loss3.sum()

        loss3.backward()
        ctrl_optimizer3.step()

            # # EM_SINGLE
            # if step and i % 10 == 0:
            #     logger.info(f'arch: {archs}')
            #     logger.info(f'Cosine_Similarity: {action_value + 0.1*EM3}, EM_Dis: {EM3}, Action_Value: {action_value}')
            #     tqdm.write(
            #         "[Epoch %d/%d] [Batch %d/%d] [ctrl1 loss: %f] [ctrl2 loss: %f] [ctrl3 loss: %f]" %
            #         (step, args.ctrl_step, i % eval_iter, eval_iter, loss1.item(), loss2.item(), loss3.item()))

            # # EM_Distributed
            # if step and i % 10 == 0:
            #     logger.info(f'arch: {archs}')
            #     logger.info(f'cos_sim: {cos_sims}, EM_Dis: {EM3}, Action_Value: {action_value}')
            #     tqdm.write(
            #         "[Epoch %d/%d] [Batch %d/%d] [ctrl1 loss: %f] [ctrl2 loss: %f] [ctrl3 loss: %f]" %
            #         (step, args.ctrl_step, i % eval_iter, eval_iter, loss1.item(), loss2.item(), loss3.item()))

        # EM + cos_sim
        tqdm.write("[Ctrl_Step %d/%d]" % (step, args.ctrl_step))
        logger.info('arch: {}'.format(archs[0].data))
        logger.info(f'Action_Value: {action_value.data}')
        logger.info(f'ctrl1_loss: {loss1.item()}, ctrl2_loss: {loss2.item()}, ctrl3_loss: {loss3.item()}')

def train_controller_rollout(args, controller1, controller2, controller3, ctrl_optimizer1, ctrl_optimizer2, ctrl_optimizer3,
                               teacher_net_b, teacher_net_dis, gen_net, dis_net, eval_loader, prev_archs, prev_hiddens, train_epoch):
    logger.info("=> train controller...")
    # writer = writer_dict['writer']
    baseline1 = None
    baseline2 = None
    baseline3 = None

    # train mode
    controller1.train()
    controller2.train()
    controller3.train()
    #
    # if prev_hiddens:
    #     # top_archs1 = prev_archs[:, :4]
    #     # top_archs2 = prev_archs[:, 4:9]
    #     # top_archs3 = prev_archs[:, 9:]
    #
    #     hxs1 = hxs2 = hxs3 = []
    #     cxs1 = cxs2 = cxs3 = []
    #     # top_archs1 = top_archs2 = top_archs3 = []
    #
    #     for (hxs, cxs, arch) in zip(prev_hiddens[0], prev_hiddens[1], prev_archs):
    #         hxs1.append(hxs[:args.hid_size])
    #         hxs2.append(hxs[args.hid_size:2 * args.hid_size])
    #         hxs3.append(hxs[2 * args.hid_size:3 * args.hid_size])
    #
    #         cxs1.append(cxs[:args.hid_size])
    #         cxs2.append(cxs[args.hid_size:2 * args.hid_size])
    #         cxs3.append(cxs[2 * args.hid_size:3 * args.hid_size])
    #
    #         # top_archs1.append(arch[:4])
    #         # top_archs2.append(arch[4:9])
    #         # top_archs3.append(arch[9:])
    #
    #     prev_hiddens1 = (hxs1, cxs1)
    #     prev_hiddens2 = (hxs2, cxs2)
    #     prev_hiddens3 = (hxs3, cxs3)
    # else:
    #     # top_archs1 = top_archs2 = top_archs3 = None
    #     prev_hiddens1 = prev_hiddens2 = prev_hiddens3 = None

    # eval mode
    gen_net.eval()

    # cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        # for i in range(eval_iter):
        ctrl_optimizer1.zero_grad()
        ctrl_optimizer2.zero_grad()
        ctrl_optimizer3.zero_grad()

        # archs1, entropies1, selected_log_probs1 = controller1.sample(args.ctrl_sample_batch, cur_top_hiddens=prev_hiddens1)
        # archs2, entropies2, selected_log_probs2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens1, cur_top_hiddens=prev_hiddens2)
        # archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens2, cur_top_hiddens=prev_hiddens3)

        # archs1, entropies1, selected_log_probs1 = controller1.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens1)
        # archs2, entropies2, selected_log_probs2 = controller2.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens2)
        # archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens3)

        archs1, entropies1, selected_log_probs1, hiddens1 = controller1.sample(args.ctrl_sample_batch, with_hidden=True)
        archs2, entropies2, selected_log_probs2, hiddens2 = controller2.sample(args.ctrl_sample_batch, with_hidden=True, prev_hiddens=hiddens1)
        archs3, entropies3, selected_log_probs3 = controller3.sample(args.ctrl_sample_batch, prev_hiddens=hiddens2)

        # cur_batch_rewards = []
        cur_batch_rewards1 = []
        cur_batch_rewards2 = []
        cur_batch_rewards3 = []
        archs = torch.cat([archs1,archs2,archs3],dim=-1)
        for (i, arch) in enumerate(archs):
            gen_net.set_arch(arch)
            # action_value, EM3, cos_sims = get_reward_block(args, gen_net, teacher_net_b, teacher_net_dis)
            # EMs, staged_EMs_stu, entire_EMs_stu, entire_dis_staged_EM_stu, entire_fake_validity_stu, staged_fake_validity_stu, action_value = get_reward_dis(args, gen_net, teacher_net_b, teacher_net_dis, eval_loader, train_epoch, arch, prev_archs)
            action_value = get_reward_rollout(args, gen_net, dis_net, hiddens1, controller2, controller3, teacher_net_b, teacher_net_dis, eval_loader, train_epoch, archs1[i], archs2[i], archs3[i], prev_archs)

            cur_batch_rewards1.append(action_value[0])
            cur_batch_rewards2.append(action_value[1])
            cur_batch_rewards3.append(action_value[2])

        cur_batch_rewards1 = torch.tensor(cur_batch_rewards1, requires_grad=False).cuda()
        cur_batch_rewards1 = cur_batch_rewards1.unsqueeze(-1) + args.entropy_coeff * entropies1

        cur_batch_rewards2 = torch.tensor(cur_batch_rewards2, requires_grad=False).cuda()
        cur_batch_rewards2 = cur_batch_rewards2.unsqueeze(-1) + args.entropy_coeff * entropies2

        cur_batch_rewards3 = torch.tensor(cur_batch_rewards3, requires_grad=False).cuda()
        cur_batch_rewards3 = cur_batch_rewards3.unsqueeze(-1) + args.entropy_coeff * entropies3

        if baseline1 is None and baseline2 is None and baseline3 is None:
            baseline1 = cur_batch_rewards1
            baseline2 = cur_batch_rewards2
            baseline3 = cur_batch_rewards3
        else:
            baseline1 = args.baseline_decay * baseline1.detach() + (1 - args.baseline_decay) * cur_batch_rewards1
            baseline2 = args.baseline_decay * baseline2.detach() + (1 - args.baseline_decay) * cur_batch_rewards2
            baseline3 = args.baseline_decay * baseline3.detach() + (1 - args.baseline_decay) * cur_batch_rewards3

        adv1 = cur_batch_rewards1 - baseline1
        adv2 = cur_batch_rewards2 - baseline2
        adv3 = cur_batch_rewards3 - baseline3

        # policy loss
        loss1 = -selected_log_probs1 * adv1  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss1 = loss1.sum()

        loss1.backward()
        ctrl_optimizer1.step()

        loss2 = -selected_log_probs2 * adv2  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss2 = loss2.sum()

        loss2.backward()
        ctrl_optimizer2.step()

        loss3 = -selected_log_probs3 * adv3  # 每个reward加上关于所有可能操作的entropy * action的概率
        loss3 = loss3.sum()

        loss3.backward()
        ctrl_optimizer3.step()

            # # EM_SINGLE
            # if step and i % 10 == 0:
            #     logger.info(f'arch: {archs}')
            #     logger.info(f'Cosine_Similarity: {action_value + 0.1*EM3}, EM_Dis: {EM3}, Action_Value: {action_value}')
            #     tqdm.write(
            #         "[Epoch %d/%d] [Batch %d/%d] [ctrl1 loss: %f] [ctrl2 loss: %f] [ctrl3 loss: %f]" %
            #         (step, args.ctrl_step, i % eval_iter, eval_iter, loss1.item(), loss2.item(), loss3.item()))

            # # EM_Distributed
            # if step and i % 10 == 0:
            #     logger.info(f'arch: {archs}')
            #     logger.info(f'cos_sim: {cos_sims}, EM_Dis: {EM3}, Action_Value: {action_value}')
            #     tqdm.write(
            #         "[Epoch %d/%d] [Batch %d/%d] [ctrl1 loss: %f] [ctrl2 loss: %f] [ctrl3 loss: %f]" %
            #         (step, args.ctrl_step, i % eval_iter, eval_iter, loss1.item(), loss2.item(), loss3.item()))

        # EM + cos_sim
        tqdm.write("[Ctrl_Step %d/%d]" % (step, args.ctrl_step))
        logger.info('arch: {}'.format(archs[0].data))
        logger.info(f'Action_Value: {action_value.data}')
        logger.info(f'ctrl1_loss: {loss1.item()}, ctrl2_loss: {loss2.item()}, ctrl3_loss: {loss3.item()}')

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

def get_reward_withEM(args, gen_net, dis_net, teacher_net_b):
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
    cell_reward = []
    # h1, h2, h3 = teacher_net_b(gen_z, tea=True)
    tea_imgs1, tea_imgs2, tea_imgs3 = transform(teacher_net_b(gen_z), args.eval_batch_size)
    imgs1, imgs2, imgs3 = gen_net(gen_z)
    stu_imgs1, stu_imgs2, stu_imgs3 = transform([imgs1, imgs2, imgs3], args.eval_batch_size)

    EM1, EM2, EM3 = dis_net(imgs3, div=True)

    cell_reward.append(torch.cosine_similarity(stu_imgs1, tea_imgs1 ,dim=1).mean() + torch.mean(EM1))
    cell_reward.append(torch.cosine_similarity(stu_imgs2, tea_imgs2, dim=1).mean() + torch.mean(EM2))
    cell_reward.append(torch.cosine_similarity(stu_imgs3, tea_imgs3, dim=1).mean() + torch.mean(EM3))
    # cells_reward.append(cell_reward)

    cell_reward = torch.tensor(cell_reward, requires_grad=False).cuda()
    # cell2_reward = torch.tensor(cell2_reward, requires_grad=False).cuda()
    # cell3_reward = torch.tensor(cell3_reward, requires_grad=False).cuda()

    return cell_reward

def get_reward_block(args, gen_net, teacher_net_b, teacher_net_dis):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    eval_iter = args.rl_num_eval_img // args.eval_batch_size
    cell_rewards = []
    EMs = []
    cos_sims = []
    # L1_Loss = torch.nn.L1Loss().cuda()

    with torch.no_grad():
        # eval mode
        gen_net = gen_net.eval()
        teacher_net_b = teacher_net_b.eval()
        teacher_net_dis = teacher_net_dis.eval()

        for i in range(eval_iter):
            cell_reward = []
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
            h, h1, h2, h3 = teacher_net_b(gen_z, tea=True)
            # tea_imgs1, tea_imgs2, tea_imgs3 = transform([h1[0], h2[0], h3[0]], args.eval_batch_size)

            imgs1, imgs2, imgs3 = gen_net(gen_z, tea=[h1,h2,h], tea_model=teacher_net_b)
            # stu_imgs1, stu_imgs2, stu_imgs3 = transform([imgs1[0], imgs2[0], imgs3[0]], args.eval_batch_size)

            _, _, fake_imgs = gen_net(gen_z)

            # distributed_EM
            stu_EM1, _, _ = teacher_net_dis(imgs1[1], div=True)
            tea_EM1, _, _ = teacher_net_dis(h1[2], div=True)

            _, stu_EM2, _ = teacher_net_dis(imgs2[1], div=True)
            _, tea_EM2, _ = teacher_net_dis(h2[2], div=True)

            _, _, stu_EM3 = teacher_net_dis(imgs3[1], div=True)
            _, _, tea_EM3 = teacher_net_dis(h3[1], div=True)

            # fake_validity = teacher_net_dis(fake_imgs)


            EM1 = torch.mean(torch.abs(tea_EM1.detach() - stu_EM1.detach()))
            EM2 = torch.mean(torch.abs(tea_EM2.detach() - stu_EM2.detach()))
            EM3 = torch.mean(torch.abs(tea_EM3.detach() - stu_EM3.detach()))

            # fake_validity = -torch.mean(fake_validity)

            cos_sim = torch.cosine_similarity(transform([fake_imgs], args.eval_batch_size)[0], transform([h3[1]], args.eval_batch_size)[0], dim=1).mean()

            EMs.append([EM1, EM2, EM3])
            cos_sims.append(cos_sim)

            # ward with EM only
            cell_reward.append(torch.exp(cos_sim + torch.pow(1.5, -EM1)))
            cell_reward.append(torch.exp(cos_sim + torch.pow(1.5, -EM2)))
            cell_reward.append(torch.exp(cos_sim + torch.pow(1.5, -EM3)))

            cell_rewards.append(cell_reward)

        cell_rewards = torch.mean(torch.tensor(cell_rewards, requires_grad=False), dim=0).cuda()
        EMs = torch.mean(torch.tensor(EMs, requires_grad=False), dim=0).cuda()
        cos_sims = torch.mean(torch.tensor(cos_sims)).cuda()

    return cell_rewards, EMs, cos_sims
    # print(cell_reward)

    # cell_rewards.append(cell_reward)

    # cell_rewards = torch.exp(torch.mean(torch.tensor(cell_rewards, requires_grad=False), dim=0)).cuda()
    # EMs = torch.mean(torch.tensor(EMs, requires_grad=False), dim=0).cuda()
    # cos_sims = torch.mean(torch.tensor(cos_sims)).cuda()

        # # reward with cos_sim and EM
        # cell_reward.append(100 * torch.cosine_similarity(stu_imgs1, tea_imgs1 ,dim=1).mean() - 0.1*EM1)
        # cell_reward.append(100 * torch.cosine_similarity(stu_imgs2, tea_imgs2, dim=1).mean() - 0.1*EM2)
        # cell_reward.append(100 * torch.cosine_similarity(stu_imgs3, tea_imgs3, dim=1).mean() - 0.1*EM3)

        # # reward with EM
        # cell1_loss = L1_Loss(stu_imgs1, tea_imgs1)
        # cell2_loss = L1_Loss(stu_imgs2, tea_imgs2)
        # cell3_loss = L1_Loss(stu_imgs3, tea_imgs3)
        #
        # cell_reward.append(10 - (EM1 + cell1_loss))
        # cell_reward.append(10 - (EM2 + 0.1 * cell2_loss))
        # cell_reward.append(10 - (0.1 * EM3 + 0.1 * cell3_loss))
        #
        # cell_reward = torch.tensor(cell_reward, requires_grad=False).cuda()

        # return cell_reward, torch.tensor([EM1, EM2, EM3], requires_grad=False), cos_sims


    # # single_EM
    # _, _, stu_EM3 = teacher_net_dis(imgs3[1], div=True)
    # _, _, tea_EM3 = teacher_net_dis(fake_img3, div=True)
    #
    # EM3 = torch.mean(torch.abs(tea_EM3.detach() - stu_EM3.detach()))
    #
    # cell_reward.append(100 * torch.cosine_similarity(stu_imgs1, tea_imgs1 ,dim=1).mean() - 0.01*EM3)
    # cell_reward.append(100 * torch.cosine_similarity(stu_imgs2, tea_imgs2, dim=1).mean() - 0.01*EM3)
    # cell_reward.append(100 * torch.cosine_similarity(stu_imgs3, tea_imgs3, dim=1).mean() - 0.01*EM3)
    #
    # cell_reward = torch.tensor(cell_reward, requires_grad=False).cuda()
    #
    # return cell_reward, EM3

def get_reward_rollout(args, gen_net, dis_net, hiddens1, hiddens2, controller2, controller3, teacher_net_b, teacher_net_dis, eval_loader, archs1, archs2, archs3, prev_archs=None):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    cell_reward1 = []
    cell_reward2 = []
    cell_reward3 = []
    action_value = []

    with torch.no_grad():
        # eval mode
        teacher_net_b = teacher_net_b.eval()
        teacher_net_dis = teacher_net_dis.eval()
        length = len(eval_loader)
        print("len_eval:{}".format(length))
        for iter_idx, (imgs, _) in enumerate(eval_loader):
            if iter_idx < 30:
                # print("arch1:{}".format(archs1))
                arch2,_,_, hiddens2 = controller2.sample(1, prev_hiddens=hiddens1, with_hidden=True)
                # print("arch2:{}".format(arch2))
                arch3 = controller3.sample(1, prev_hiddens=hiddens2)[0][0]
                # print("arch3:{}".format(arch3))
                archs = torch.cat([archs1, arch2[0], arch3], dim=-1)
                # print("iter_idx:{}, archs:{}".format(iter_idx, archs))

                real_imgs = imgs.type(torch.cuda.FloatTensor)
                # print(real_imgs.shape)
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim)))

                gen_net.set_arch(archs)
                gen_net = gen_net.eval()
                _, _, fake_imgs3 = gen_net(gen_z)

                # fake_validity3 = teacher_net_dis(fake_imgs3)
                # real_validity3 = teacher_net_dis(real_imgs)
                fake_validity3 = dis_net(fake_imgs3)
                real_validity3 = dis_net(real_imgs)

                roll_out1 = torch.mean(real_validity3) - torch.mean(fake_validity3)
                # roll_out1 = 10*(1-torch.sigmoid(roll_out1))

                cell_reward1.append(roll_out1)

            elif iter_idx >= 30 and iter_idx < 60:
                # arch1 = controller1.sample(1)[0][0]
                arch3 = controller3.sample(1, prev_hiddens=hiddens2)[0][0]
                archs = torch.cat([archs1, archs2, arch3], dim=-1)
                # print("iter_idx:{}, archs:{}".format(iter_idx, archs))

                real_imgs = imgs.type(torch.cuda.FloatTensor)
                # print(real_imgs.shape)
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim)))

                gen_net.set_arch(archs)
                gen_net = gen_net.eval()
                _, _, fake_imgs3 = gen_net(gen_z)

                # fake_validity3 = teacher_net_dis(fake_imgs3)
                # real_validity3 = teacher_net_dis(real_imgs)
                fake_validity3 = dis_net(fake_imgs3)
                real_validity3 = dis_net(real_imgs)

                roll_out2 = torch.mean(real_validity3) - torch.mean(fake_validity3)
                # roll_out2 = 10 * (1 - torch.sigmoid(roll_out2))

                cell_reward2.append(roll_out2)

            elif iter_idx >= 60 and iter_idx < 90:
                # arch1 = controller1.sample(1)[0][0]
                # arch2 = controller2.sample(1)[0][0]
                archs = torch.cat([archs1, archs2, archs3], dim=-1)
                # print("iter_idx:{}, archs:{}".format(iter_idx, archs))

                real_imgs = imgs.type(torch.cuda.FloatTensor)
                # print(real_imgs.shape)
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim)))

                gen_net.set_arch(archs)
                gen_net = gen_net.eval()
                _, _, fake_imgs3 = gen_net(gen_z)

                # fake_validity3 = teacher_net_dis(fake_imgs3)
                # real_validity3 = teacher_net_dis(real_imgs)
                fake_validity3 = dis_net(fake_imgs3)
                real_validity3 = dis_net(real_imgs)

                roll_out3 = torch.mean(real_validity3) - torch.mean(fake_validity3)
                # roll_out3 = 10 * (1 - torch.sigmoid(roll_out3))

                cell_reward3.append(roll_out3)

            else:
                break

        cell_reward1 = torch.mean(torch.tensor(cell_reward1, requires_grad=False), dim=0)
        cell_reward2 = torch.mean(torch.tensor(cell_reward2, requires_grad=False), dim=0)
        cell_reward3 = torch.mean(torch.tensor(cell_reward3, requires_grad=False), dim=0)

        action_value = torch.tensor([(1 - torch.sigmoid(cell_reward1)), - torch.sigmoid(cell_reward2), (1 - torch.sigmoid(cell_reward3))], requires_grad=False).cuda()

        return action_value


def get_reward_dis(args, gen_net, teacher_net_b, teacher_net_dis, eval_loader, train_epoch, arch=None, prev_archs=None):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval_iter = args.rl_num_eval_img // args.eval_batch_size
    # cell_rewards = []
    EMs = []
    cos_sims1 = []
    cos_sims2 = []
    cos_sims3 = []
    staged_EMs_stu = []
    entire_EMs_stu = []
    entire_fake_validity_stu = []
    staged_fake_validity_stu = []
    entire_dis_staged_EM_stu = []
    action_value = []

    with torch.no_grad():
        # eval mode
        gen_net = gen_net.eval()
        teacher_net_b = teacher_net_b.eval()
        teacher_net_dis = teacher_net_dis.eval()

        # arch1 = arch[:4].unsqueeze(0).type(torch.cuda.FloatTensor)
        # arch2 = arch[4:9].unsqueeze(0).type(torch.cuda.FloatTensor)
        # arch3 = arch[9:].unsqueeze(0).type(torch.cuda.FloatTensor)
        #
        # if prev_archs:
        #     for prev_arch in prev_archs:
        #
        #         prev_arch1 = prev_arch[:4].unsqueeze(0).type(torch.cuda.FloatTensor)
        #         prev_arch2 = prev_arch[4:9].unsqueeze(0).type(torch.cuda.FloatTensor)
        #         prev_arch3 = prev_arch[9:].unsqueeze(0).type(torch.cuda.FloatTensor)
        #
        #         # cos_sims1.append(torch.cosine_similarity(arch1, prev_arch1))
        #         # cos_sims2.append(torch.cosine_similarity(arch2, prev_arch2))
        #         # cos_sims3.append(torch.cosine_similarity(arch3, prev_arch3))
        #
        #         cos_sims1.append(torch.sqrt(1 + torch.cosine_similarity(arch1, prev_arch1)))
        #         cos_sims2.append(torch.sqrt(1 + torch.cosine_similarity(arch2, prev_arch2)))
        #         cos_sims3.append(torch.sqrt(1 + torch.cosine_similarity(arch3, prev_arch3)))
        #
        #     cos_sims1 = torch.mean(torch.tensor(cos_sims1, requires_grad=False))
        #     cos_sims2 = torch.mean(torch.tensor(cos_sims2, requires_grad=False))
        #     cos_sims3 = torch.mean(torch.tensor(cos_sims3, requires_grad=False))
        # else:
        #     cos_sims1 = cos_sims2 = cos_sims3 = torch.tensor(1).type(torch.cuda.FloatTensor)

        for iter_idx, (imgs, _) in enumerate(eval_loader):

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            # print(real_imgs.shape)
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim)))

            h, h1, h2, h3 = teacher_net_b(gen_z, tea=True)
            # tea_imgs1, tea_imgs2, tea_imgs3 = transform([h1[0], h2[0], h3[0]], args.eval_batch_size)

            imgs1, imgs2, imgs3 = gen_net(gen_z, tea=[h1,h2,h], tea_model=teacher_net_b)
            # stu_imgs1, stu_imgs2, stu_imgs3 = transform([imgs1[0], imgs2[0], imgs3[0]], args.eval_batch_size)

            fake_imgs1, fake_imgs2, fake_imgs3 = gen_net(gen_z)

            # distributed_EM
            stu_EM1, _, _ = teacher_net_dis(imgs1[1], div=True)
            tea_EM1, _, _ = teacher_net_dis(h1[2], div=True)

            _, stu_EM2, _ = teacher_net_dis(imgs2[1], div=True)
            _, tea_EM2, _ = teacher_net_dis(h2[2], div=True)

            _, _, stu_EM3 = teacher_net_dis(imgs3[1], div=True)
            _, _, tea_EM3 = teacher_net_dis(h3[1], div=True)

            #
            fake_validity1, fake_validity2, fake_validity3 = teacher_net_dis(fake_imgs3, div=True)
            real_validity1, real_validity2, real_validity3 = teacher_net_dis(real_imgs, div=True)

            # fake_validity1, _, _ = teacher_net_dis(fake_imgs1, div=True)
            # _, fake_validity2, _ = teacher_net_dis(fake_imgs2, div=True)
            # _, _, fake_validity3 = teacher_net_dis(fake_imgs3, div=True)

            staged_dis_fake_validity1 = -torch.mean(fake_validity1)
            staged_dis_fake_validity2 = -torch.mean(fake_validity2)
            staged_dis_fake_validity3 = -torch.mean(fake_validity3)

            entire_dis_fake_validity1 = -torch.mean(teacher_net_dis(fake_imgs1))
            entire_dis_fake_validity2 = -torch.mean(teacher_net_dis(fake_imgs2))
            entire_dis_fake_validity3 = -torch.mean(teacher_net_dis(fake_imgs3))

            EM1 = torch.abs(torch.mean(tea_EM1.detach() - stu_EM1.detach()))
            EM2 = torch.abs(torch.mean(tea_EM2.detach() - stu_EM2.detach()))
            EM3 = torch.abs(torch.mean(tea_EM3.detach() - stu_EM3.detach()))

            # EM_stu1 = torch.abs(torch.mean(real_validity1) - torch.mean(fake_validity1))
            # EM_stu2 = torch.abs(torch.mean(real_validity2) - torch.mean(fake_validity2))
            # EM_stu3 = torch.abs(torch.mean(real_validity3) - torch.mean(fake_validity3))

            staged_EM_stu1 = torch.mean(real_validity1) + staged_dis_fake_validity1
            staged_EM_stu2 = torch.mean(real_validity2) + staged_dis_fake_validity2
            staged_EM_stu3 = torch.mean(real_validity3) + staged_dis_fake_validity3

            entire_EM_stu1 = torch.mean(real_validity3) + entire_dis_fake_validity1
            entire_EM_stu2 = torch.mean(real_validity3) + entire_dis_fake_validity2
            entire_EM_stu3 = torch.mean(real_validity3) + entire_dis_fake_validity3

            entire_dis_staged_EM_stu1 = torch.mean(real_validity1) + entire_dis_fake_validity1
            entire_dis_staged_EM_stu2 = torch.mean(real_validity2) + entire_dis_fake_validity2
            entire_dis_staged_EM_stu3 = torch.mean(real_validity3) + entire_dis_fake_validity3

            EMs.append([EM1, EM2, EM3])
            # cos_sims.append(cos_sim)
            staged_EMs_stu.append([staged_EM_stu1, staged_EM_stu2, staged_EM_stu3])
            entire_fake_validity_stu.append([entire_dis_fake_validity1, entire_dis_fake_validity2, entire_dis_fake_validity3])
            entire_EMs_stu.append([entire_EM_stu1, entire_EM_stu2, entire_EM_stu3])
            staged_fake_validity_stu.append([staged_dis_fake_validity1, staged_dis_fake_validity2, staged_dis_fake_validity3])
            entire_dis_staged_EM_stu.append([entire_dis_staged_EM_stu1, entire_dis_staged_EM_stu2, entire_dis_staged_EM_stu3])

            # ward with EM only
            cell_reward = []

            # cell_reward.append(torch.exp(-cos_sims1) + torch.exp(-EM1) + torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(-cos_sims2) + torch.exp(-EM2) + torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(-cos_sims3) + torch.exp(-EM3) + torch.exp(-EM_stu))

            # #
            # cell_reward.append(torch.exp(torch.pow(cos_sims1, 1)) + torch.exp(-EM1))
            # cell_reward.append(torch.exp(torch.pow(cos_sims2, 1)) + torch.exp(-EM2))
            # cell_reward.append(torch.exp(torch.pow(cos_sims3, 1)) + torch.exp(-EM3))

            # cell_reward.append(torch.exp(torch.pow(cos_sims1, -1)) + torch.exp(-EM1) + 0.5 * torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(torch.pow(cos_sims2, -1)) + torch.exp(-EM2) + 0.5 * torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(torch.pow(cos_sims3, -1)) + torch.exp(-EM3) + 0.5 * torch.exp(-EM_stu))

            # cell_reward.append(torch.exp(-EM1) + 0.5 * torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(-EM2) + 0.5 * torch.exp(-EM_stu))
            # cell_reward.append(torch.exp(-EM3) + 0.5 * torch.exp(-EM_stu))

            # cell_reward.append(torch.pow(EM1, -1))
            # cell_reward.append(torch.pow(EM2, -1))
            # cell_reward.append(torch.pow(EM3, -1))

            # cell_rewards.append(cell_reward)

        # cell_rewards = torch.mean(torch.tensor(cell_rewards, requires_grad=False), dim=0).cuda()
        EMs = torch.mean(torch.tensor(EMs, requires_grad=False), dim=0)
        staged_EMs_stu = torch.mean(torch.tensor(staged_EMs_stu, requires_grad=False), dim=0)
        entire_EMs_stu = torch.mean(torch.tensor(entire_EMs_stu, requires_grad=False), dim=0)
        entire_fake_validity_stu = torch.mean(torch.tensor(entire_fake_validity_stu, requires_grad=False), dim=0)
        staged_fake_validity_stu = torch.mean(torch.tensor(staged_fake_validity_stu, requires_grad=False), dim=0)
        entire_dis_staged_EM_stu = torch.mean(torch.tensor(entire_dis_staged_EM_stu, requires_grad=False), dim=0)
        # action_value = entire_fake_validity_stu.cuda()
        # action_value[0] *= 10
        # entire_dis_staged_EM_stu = torch.mean(torch.tensor(entire_dis_staged_EM_stu, requires_grad=False), dim=0)

        # action_value = [1 - torch.sigmoid(entire_dis_fake_validity3), 1 - torch.sigmoid(entire_dis_fake_validity3),
        #                 1 - torch.sigmoid(entire_dis_fake_validity3)]
        # action_value = [10*(1-torch.sigmoid(entire_EM_stu3)), 10*(1-torch.sigmoid(entire_EM_stu3)), 10*(1-torch.sigmoid(entire_EM_stu3))]
        if train_epoch >= 20:
            action_value = [10*(1 - 0.01 * staged_fake_validity_stu[0]), 10*(1 - 0.1 * staged_fake_validity_stu[1]), 10*(1 - 0.1 * staged_fake_validity_stu[2])]
        else:
            action_value = [(1 - 0.01 * staged_fake_validity_stu[0]), (1 - 0.1 * staged_fake_validity_stu[1]), (1 - 0.1 * staged_fake_validity_stu[2])]

        action_value = torch.tensor(action_value, requires_grad=False).cuda()

        # cell_rewards = (torch.pow(EMs, -1) + torch.pow(EMs_stu, -1)).cuda()
        # cos_sims = torch.mean(torch.tensor(cos_sims)).cuda()

    return EMs, staged_EMs_stu, entire_EMs_stu, entire_dis_staged_EM_stu, entire_fake_validity_stu, staged_fake_validity_stu, action_value
    # return cell_rewards, EMs, [cos_sims1, cos_sims2, cos_sims3], staged_EMs_stu

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

def get_topk_arch_hidden(args, controller1, controller2, controller3, gen_net):
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
    controller1.eval()
    controller2.eval()
    controller3.eval()

    archs1 = controller1.sample(args.num_candidate)[0]
    archs2 = controller2.sample(args.num_candidate)[0]
    archs3 = controller3.sample(args.num_candidate)[0]

    # hxs, cxs = hiddens
    archs = torch.cat([archs1,archs2,archs3],dim=-1)
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


def get_topk_arch_with_hidden(args, controller1, controller2, controller3, gen_net, prev_archs, prev_hiddens):
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
    controller1.eval()
    controller2.eval()
    controller3.eval()

    with torch.no_grad():
        if prev_hiddens:
            # top_archs1 = prev_archs[:, :4]
            # top_archs2 = prev_archs[:, 4:9]
            # top_archs3 = prev_archs[:, 9:]

            hxs1 = hxs2 = hxs3 = []
            cxs1 = cxs2 = cxs3 = []
            top_archs1 = top_archs2 = top_archs3 = []

            for (hxs, cxs, arch) in zip(prev_hiddens[0], prev_hiddens[1], prev_archs):
                hxs1.append(hxs[:args.hid_size])
                hxs2.append(hxs[args.hid_size:2 * args.hid_size])
                hxs3.append(hxs[2 * args.hid_size:3 * args.hid_size])

                cxs1.append(cxs[:args.hid_size])
                cxs2.append(cxs[args.hid_size:2 * args.hid_size])
                cxs3.append(cxs[2 * args.hid_size:3 * args.hid_size])

                top_archs1.append(arch[:4])
                top_archs2.append(arch[4:9])
                top_archs3.append(arch[9:])

            prev_hiddens1 = (hxs1, cxs1)
            prev_hiddens2 = (hxs2, cxs2)
            prev_hiddens3 = (hxs3, cxs3)
        else:
            top_archs1 = top_archs2 = top_archs3 = None
            prev_hiddens1 = prev_hiddens2 = prev_hiddens3 = None

    archs1, _, _, hiddens1 = controller1.sample(args.num_candidate, with_hidden=True, prev_archs=top_archs1, prev_hiddens=prev_hiddens1)
    archs2, _, _, hiddens2 = controller2.sample(args.num_candidate, with_hidden=True, prev_archs=top_archs2, prev_hiddens=prev_hiddens2)
    archs3, _, _, hiddens3 = controller3.sample(args.num_candidate, with_hidden=True, prev_archs=top_archs3, prev_hiddens=prev_hiddens3)

    hxs1, cxs1 = hiddens1
    hxs2, cxs2 = hiddens2
    hxs3, cxs3 = hiddens3

    hxs = torch.cat([hxs1, hxs2, hxs3], dim=1)
    cxs = torch.cat([cxs1, cxs2, cxs3], dim=1)
    archs = torch.cat([archs1, archs2, archs3], dim=-1)
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx])
        is_score = get_IS(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score

    # topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]

    topk_arch_idx_perf = {k: v for k, v in arch_idx_perf_table.items() if v >= 5.5}
    topk_arch_idx_perf = sorted(topk_arch_idx_perf.items(), key=operator.itemgetter(1))[::-1]

    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))
    # print("top_hxs_length:{}, length_of_topk_hxs[0]:{}".format(len(topk_hxs), len(topk_hxs[0])))

    return topk_archs, (topk_hxs, topk_cxs)

def get_top5_arch_with_hidden(args, controller1, controller2, controller3, gen_net, prev_archs, prev_hiddens):
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
    controller1.eval()
    controller2.eval()
    controller3.eval()

    with torch.no_grad():
        if prev_hiddens:
            prev_hxs1 = []
            prev_hxs2 = []
            prev_hxs3 = []

            prev_cxs1 = []
            prev_cxs2 = []
            prev_cxs3 = []
            hxs, cxs = prev_hiddens
            # print("hxs:{}".format(hxs))

            for (hx, cx) in zip(hxs, cxs):
                prev_hxs1.append(hx[0:args.hid_size])
                prev_hxs2.append(hx[args.hid_size:args.hid_size * 2])
                prev_hxs3.append(hx[args.hid_size * 2:args.hid_size * 3])

                prev_cxs1.append(cx[0:args.hid_size])
                prev_cxs2.append(cx[args.hid_size:args.hid_size * 2])
                prev_cxs3.append(cx[args.hid_size * 2:args.hid_size * 3])

            prev_hiddens1 = (prev_hxs1, prev_cxs1)
            prev_hiddens2 = (prev_hxs2, prev_cxs2)
            prev_hiddens3 = (prev_hxs3, prev_cxs3)

            # archs1, _,_, hiddens1 = controller1.sample(args.num_candidate, with_hidden=True)
            # archs2, _,_, hiddens2 = controller2.sample(args.num_candidate, prev_hiddens=prev_hiddens1, with_hidden=True)
            # archs3, _,_, hiddens3 = controller3.sample(args.num_candidate, prev_hiddens=prev_hiddens2, with_hidden=True)

            archs1, _,_, hiddens1 = controller1.sample(args.num_candidate, with_hidden=True, cur_top_hiddens=prev_hiddens1)
            archs2, _,_, hiddens2 = controller2.sample(args.num_candidate, prev_hiddens=hiddens1, with_hidden=True, cur_top_hiddens=prev_hiddens2)
            archs3, _,_, hiddens3 = controller3.sample(args.num_candidate, prev_hiddens=hiddens2, with_hidden=True, cur_top_hiddens=prev_hiddens3)

        else:
            archs1, _, _, hiddens1 = controller1.sample(args.num_candidate, with_hidden=True)
            archs2, _, _, hiddens2 = controller2.sample(args.num_candidate, with_hidden=True, prev_hiddens=hiddens1)
            archs3, _, _, hiddens3 = controller3.sample(args.num_candidate, with_hidden=True, prev_hiddens=hiddens2)

        hxs1, cxs1 = hiddens1
        hxs2, cxs2 = hiddens2
        hxs3, cxs3 = hiddens3

        hxs = torch.cat([hxs1, hxs2, hxs3], dim=1)
        cxs = torch.cat([cxs1, cxs2, cxs3], dim=1)
        archs = torch.cat([archs1, archs2, archs3], dim=-1)

        arch_idx_perf_table = {}
        for arch_idx in range(len(archs)):
            logger.info(f'arch: {archs[arch_idx]}')
            gen_net.set_arch(archs[arch_idx])
            is_score = get_IS(args, gen_net, args.rl_num_eval_img)
            logger.info(f'get Inception score of {is_score}')
            arch_idx_perf_table[arch_idx] = is_score

        topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]

        # topk_arch_idx_perf = {k: v for k, v in arch_idx_perf_table.items() if v >= 5.5}
        # topk_arch_idx_perf = sorted(topk_arch_idx_perf.items(), key=operator.itemgetter(1))[::-1]

        topk_archs = []
        topk_hxs = []
        topk_cxs = []
        logger.info(f'top{args.topk} archs:')
        for arch_idx_perf in topk_arch_idx_perf:
            logger.info(arch_idx_perf)
            arch_idx = arch_idx_perf[0]
            topk_archs.append(archs[arch_idx])
            topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
            topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))
    # print("top_hxs_length:{}, length_of_topk_hxs[0]:{}".format(len(topk_hxs), len(topk_hxs[0])))

    return topk_archs, (topk_hxs, topk_cxs)


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





