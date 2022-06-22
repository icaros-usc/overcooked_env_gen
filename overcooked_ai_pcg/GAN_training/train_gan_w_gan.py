"""This file is for pure gan experiment."""

import argparse
import math
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch import nn
from torch.autograd import Variable

from overcooked_ai_pcg.GAN_training import dcgan
import os
import json
from overcooked_ai_pcg.helper import read_in_training_data, obj_types, plot_err, save_gan_param
from overcooked_ai_py import LAYOUTS_DIR
from overcooked_ai_pcg import GAN_TRAINING_DIR


def run(
    nz,
    ngf,
    ndf,
    batch_size,
    niter,
    lrD,
    lrG,
    beta1,
    cuda,
    ngpu,
    gpu_id,
    path_netG,
    path_netD,
    clamp_lower,
    clamp_upper,
    n_extra_layers,
    gan_experiment,
    adam,
    seed,
    lvl_data,
    save_length,
    map_size,
    size_version,
):

    lvl_size = None
    sub_dir = None
    if size_version == "small":
        lvl_size = (6, 9)
        sub_dir = "train_gan_small"
    elif size_version == "large":
        lvl_size = (10, 15)
        sub_dir = "train_gan_large"

    lvl_data = os.path.join(LAYOUTS_DIR, sub_dir)

    os.makedirs(gan_experiment, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    X = read_in_training_data(lvl_data, sub_dir)
    z_dims = len(obj_types)
    print('x_shape',
          X.shape)  # shape must be num_lvls x lvl_height x lvl_width

    # set up input to the GAN, size is batch_size x z_dim x map_size x map_size
    num_batches = X.shape[0] / batch_size
    X_onehot = np.eye(z_dims, dtype='uint8')[X]
    X_onehot = np.rollaxis(X_onehot, 3, 1)
    X_train = np.zeros((X.shape[0], z_dims, map_size, map_size))
    X_train[:, 1, :, :] = 1.0
    X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

    # apply a initialization function to each module of the network
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    netG.apply(weights_init)
    if path_netG != '':
        netG.load_state_dict(torch.load(path_netG))

    netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)
    if path_netD != '':
        netD.load_state_dict(torch.load(path_netD))

    input = torch.FloatTensor(batch_size, z_dims, map_size, map_size)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)  # used for trainng
    fixed_noise = torch.FloatTensor(batch_size, nz, 1,
                                    1).normal_(0, 1)  # used for testing

    # use Binary Cross Entropy loss
    criterion = nn.BCELoss()

    one = torch.FloatTensor([1])
    mone = one * -1

    # get current device
    device = torch.device("cuda:{}".format(gpu_id) if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")

    # move data to GPU
    if cuda:
        netD.cuda(gpu_id)
        netG.cuda(gpu_id)
        input = input.cuda(gpu_id)
        noise, fixed_noise = noise.cuda(gpu_id), fixed_noise.cuda(gpu_id)
        one, mone = one.cuda(gpu_id), mone.cuda(gpu_id)

    # setup optimizer
    if adam:
        optimizerD = optim.Adam(netD.parameters(),
                                lr=lrD,
                                betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(),
                                lr=lrG,
                                betas=(beta1, 0.999))
        print("Using ADAM")
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=lrG)

    # record errors for plotting
    average_errG_log = []
    average_errD_fake_log = []
    average_errD_real_log = []
    average_errD_log = []
    average_D_x_log = []
    average_D_G_z1_log = []
    average_D_G_z2_log = []

    # main trainng loop
    for epoch in range(niter):
        X_train = X_train[torch.randperm(len(X_train))]

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        i = 0
        total_errD_fake = 0
        total_errD_real = 0
        total_errG = 0
        # total_errD = 0
        # total_D_x = 0
        # total_D_G_z1 = 0
        # total_D_G_z2 = 0
        Diters = 20
        j = 0
        while i < num_batches:  # len(dataloader):
            while j < Diters and i < num_batches:
                j += 1
                netD.zero_grad()
                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                data = X_train[i * batch_size:(i + 1) * batch_size]

                i += 1

                if cuda:
                    real_cpu = torch.FloatTensor(data).cuda(gpu_id)
                else:
                    real_cpu = torch.FloatTensor(data)

                input.resize_as_(real_cpu).copy_(
                    real_cpu)  # copy data to input

                # W-loss
                errD_real = netD(input)
                errD_real.backward(one)
                total_errD_real += errD_real.item()

                # train with fake
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                fake = netG(noise)
                errD_fake = netD(fake.detach())
                errD_fake.backward(mone)
                total_errD_fake += errD_fake.item()

                optimizerD.step()

            # # construct real and fake labels
            # curr_batch_size = input.size(0)
            # real_label = torch.full((curr_batch_size,), 1, device=device)
            # fake_label = torch.full((curr_batch_size,), 0, device=device)

            # # compute gradient of real input image
            # # D maximize the likelihood of real image being real
            # output = netD(input).view(-1)
            # errD_real = criterion(output, real_label)
            # errD_real.backward()
            # total_errD_real += errD_real.item()
            # D_x = output.mean().item()
            # total_D_x += D_x

            # # compute gradient of fake input image from G
            # # D minimize the likelihood of the fake image being real
            # noise.resize_(curr_batch_size, nz, 1, 1).normal_(0, 1)
            # fake = netG(noise)
            # output = netD(fake.detach()).view(-1)
            # errD_fake = criterion(output, fake_label)
            # errD_fake.backward()
            # D_G_z1 = output.mean().item()
            # errD = errD_real + errD_fake

            # total_errD_fake += errD_fake.item()
            # total_D_G_z1 += D_G_z1
            # total_errD += errD.item()

            # optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # # G maximize the likelihood of the fake image being real
            # output = netD(fake).view(-1)
            # errG = criterion(output, real_label)
            # errG.backward()
            # D_G_z2 = output.mean().item()

            # total_D_G_z2 += D_G_z2
            # total_errG += errG.item()
            # optimizerG.step()

            # w-loss
            errG = netD(fake)
            errG.backward(one)
            total_errG += errG.item()
            optimizerG.step()

            i += 1

        average_errG = total_errG / num_batches
        average_errD_fake = total_errD_fake / num_batches
        average_errD_real = total_errD_real / num_batches
        # average_errD = total_errD / num_batches
        # average_D_x = total_D_x / num_batches
        # average_D_G_z1 = total_D_G_z1 / num_batches
        # average_D_G_z2 = total_D_G_z2 / num_batches

        average_errG_log.append(average_errG)
        average_errD_fake_log.append(average_errD_fake)
        average_errD_real_log.append(average_errD_real)
        # average_errD_log.append(average_errD)
        # average_D_x_log.append(average_D_x)
        # average_D_G_z1_log.append(average_D_G_z1)
        # average_D_G_z2_log.append(average_D_G_z2)

        # print('[%d/%d] Loss_G: %f Loss_D: %f Loss_D_real: %f Loss_D_fake %f D(x) %f D(G(z)) %f / %f'
        #       % (epoch, niter, average_errG, average_errD, average_errD_real, average_errD_fake,
        #          average_D_x, average_D_G_z1, average_D_G_z2))

        print(
            '[%d/%d] Loss_G: %f Loss_D_real: %f Loss_D_fake %f' %
            (epoch, niter, average_errG, average_errD_real, average_errD_fake))

        # use trained G to generate fake levels from fixed noise vector once in a while
        if epoch % save_length == save_length - 1 or epoch == niter - 1:
            netG.eval()
            with torch.no_grad():
                fake = netG(fixed_noise)
                im = fake.cpu().numpy()[:, :, :lvl_size[0], :lvl_size[1]]
                im = np.argmax(im, axis=1)
            with open(
                    '{0}/fake_level_epoch_{1}_{2}.json'.format(
                        gan_experiment, epoch, seed), 'w') as f:
                f.write(json.dumps(im[0].tolist()))
            torch.save(
                netG.state_dict(),
                '{0}/netG_epoch_{1}_{2}.pth'.format(gan_experiment, epoch,
                                                    seed))

    # save Generator constructor params for generating levels later
    G_params = {
        'isize': map_size,
        'nz': nz,
        'nc': len(obj_types),
        'ngf': ngf,
        'ngpu': ngpu,
        'n_extra_layers': n_extra_layers
    }
    save_gan_param(G_params)

    # plot the stats
    plot_err(average_errG_log, average_errD_log, average_errD_fake_log,
             average_errD_real_log, average_D_x_log, average_D_G_z1_log,
             average_D_G_z2_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz',
                        type=int,
                        default=32,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='input batch size')
    parser.add_argument('--niter',
                        type=int,
                        default=10000,
                        help='number of epochs to train for')
    parser.add_argument('--lrD',
                        type=float,
                        default=0.00001,
                        help='learning rate for Critic')
    parser.add_argument('--lrG',
                        type=float,
                        default=0.00001,
                        help='learning rate for Generator')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu',
                        type=int,
                        default=1,
                        help='number of GPUs to use')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='the id of the gpu to use')
    parser.add_argument('--path_netG',
                        default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--path_netD',
                        default='',
                        help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--n_extra_layers',
                        type=int,
                        default=0,
                        help='Number of extra layers on gen and disc')
    parser.add_argument('--gan_experiment',
                        help='Where to store samples and models',
                        default=GAN_TRAINING_DIR)
    parser.add_argument('--adam',
                        action='store_true',
                        help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--seed',
                        type=int,
                        default=999,
                        help='random seed for reproducibility')
    parser.add_argument('--lvl_data',
                        help='Path to the human designed levels.',
                        default=LAYOUTS_DIR)
    parser.add_argument('--save_length',
                        type=int,
                        default=100,
                        help='Length of save point')
    parser.add_argument('--map_size',
                        type=int,
                        default=16,
                        help='Size of the initial layer of feature map of D')
    parser.add_argument(
        '--size_version',
        type=str,
        default="small",
        help='Size of the level. Small for (6, 9), large for (10, 15)')
    opt = parser.parse_args()

    run(opt.nz, opt.ngf, opt.ndf, opt.batch_size, opt.niter, opt.lrD, opt.lrG,
        opt.beta1, opt.cuda, opt.ngpu, opt.gpu_id, opt.path_netG,
        opt.path_netD, opt.clamp_lower, opt.clamp_upper, opt.n_extra_layers,
        opt.gan_experiment, opt.adam, opt.seed, opt.lvl_data, opt.save_length,
        opt.map_size, opt.size_version)
