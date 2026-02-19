import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_features, size_w=680, size_h=480, n_scale=3, do_norm=False):
        """
        Constructor
        :param num_features: depth of feature maps
        :param n_scale: number of downscaling of images within the discriminator (default=3)
        :param size_w: width of the input image (default=680)
        :param size_h: height of the input image (default=480)
        :param do_norm: apply instance normalization in the discriminator (default=False)
        """
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.n_scale = n_scale
        self.do_norm = do_norm

        self.modules_logs, self.modules_features = self.init_architecture(size_w, size_h)

    def init_architecture(self, size_w, size_h):
        """
        Initializes architecture of the discriminator depending on the dimensionality of the input images
        :param size_w: width of the input image
        :param size_h: height of the input image
        :return: ModuleLists of fully-connected and convolutional layers
        """
        if (size_w == 620 or size_w == 480) and size_h == 480:
            n_dis = 6
            max_channels = 1024
        elif (size_w == 160 or size_w == 120) and size_h == 120:
            n_dis = 4
            max_channels = 256
        else:
            raise ValueError("image input dimension not supported")

        modules_logs = nn.ModuleList()
        modules_features = nn.ModuleList()

        num_channels = 3
        for scale in range(self.n_scale):
            current_num_features = self.num_features
            layers = []
            layers += [self.conv(num_channels, current_num_features,
                                 kernel_size=4, stride=2, pad=1, pad_type='reflect')]
            if self.do_norm:
                layers += [nn.InstanceNorm2d(current_num_features, affine=True)]
            for i in range(1, n_dis):
                layers += [self.conv(current_num_features, current_num_features * 2,
                                     kernel_size=4, stride=2, pad=1, pad_type='reflect')]
                if self.do_norm:
                    layers += [nn.InstanceNorm2d(current_num_features * 2, affine=True)]
                if current_num_features < max_channels:
                    current_num_features = current_num_features * 2

            modules_features += [nn.Sequential(*layers)]

            last_conv_height = self.compute_final_conv_layer_dim(size_h, n_dis, scale)
            last_conv_width = self.compute_final_conv_layer_dim(size_w, n_dis, scale)
            modules_logs += [nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(current_num_features * 2 * last_conv_height * last_conv_width, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1)
            )]

        return modules_logs, modules_features

    def forward(self, x):
        d_logit, _ = self.predictions(x)
        # Computes the prediction for the Hinge GAN loss
        predictions = torch.mean(torch.stack(d_logit), dim=0)
        # Computes the prediction for the standard GAN loss
        predictions = torch.sigmoid(predictions)
        return predictions

    def predictions(self, im):
        d_logit = []
        features = []
        for scale in range(self.n_scale):
            x = self.modules_features[scale](im)
            features.append(x)
            d_logit.append(self.modules_logs[scale](x))

            if scale != self.n_scale - 1:
                im = nn.functional.avg_pool2d(im, kernel_size=3, stride=2, padding=1)

        return d_logit, features

    @staticmethod
    def compute_final_conv_layer_dim(dim_len, n_dis, scale):
        """
        Based on the respective height and width of the input images we compute the height and width after the last
        convolution, depending on the scale and n_dis.
        :param dim_len:
        :param n_dis:
        :param scale:
        :return:
        """
        # dim_len /= 2 defines the respective maximum height and width of the input images after one convolution.
        dim_len /= 2
        n_dis -= 1
        return int(dim_len / (2 ** (n_dis + scale)))

    @staticmethod
    def conv(in_channels, out_channels, kernel_size=4, stride=2, pad=0, pad_type='zero', use_bias=True):
        padding = (pad, pad, pad, pad)
        if pad_type == 'reflect':
            layer = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            layer = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            layer = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % pad_type)
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=0, bias=use_bias)
        conv_layer.weight = nn.init.kaiming_normal_(conv_layer.weight)
        return nn.Sequential(layer, conv_layer, nn.LeakyReLU(0.2, inplace=True))
