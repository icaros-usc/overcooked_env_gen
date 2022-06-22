import torch
import torch.nn as nn
import torch.nn.parallel


class DCGAN_D(nn.Module):
    """
    Discriminator DCGAN.
    """
    def __init__(self,
                 isize,
                 nz,
                 nc,
                 ndf,
                 ngpu,
                 n_extra_layers=0,
                 algo="w_gan"):
        """
        isize: size of input image
        nz: size of latent z vector
        nc: total number of objects in the environment
        ndf: number of output channels of initial conv2d layer
        ngpu: number of GPUs
        n_extra_layers: number of extra layers with out_channels to be ndf to add
        algo: algorithm used to train the GAN.
              "w_gan" for WGAN algorithm.
              "vanilla" for vanilla training technique.

        Note:
        input to the GAN is nc x isize x isize
        output from the GAN is the likehood of the image being real
        """
        super(DCGAN_D, self).__init__()
        self.algo = algo
        self.ngpu = ngpu
        self.nz = nz
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:conv:{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:relu:{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # add extra layers with out_channels set to ndf
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        # add more conv2d layers with exponentially more out_channels
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))

        if self.algo == "vanilla":
            # sigmoid to keep output in range [0, 1]
            main.add_module('final:sigmoid', nn.Sigmoid())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        if self.algo == "w_gan":
            output = output.mean(0)
            return output.view(1)

        elif self.algo == "vanilla":
            return output


class DCGAN_G(nn.Module):
    """
    Generator DCGAN
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        """
        isize: size of input image
        nz: size of latent z vector
        nc: total number of objects in the environment
        ngf: number of output channels of initial conv2d layer
        ngpu: number of GPUs
        n_extra_layers: number of extra layers with out_channels to be ngf to add

        Note: input is a latent vector of size nz
        """
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                'pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc), nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output