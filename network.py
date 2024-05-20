from function import  *
from functools import reduce, partial
from restormer_arch import *
from math import floor, log2

'''----------------------TPIM-----------------------'''
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# UNet with residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            # hard to converge with out batch or instance norm
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out
        # return self.relu(x + self.block(x))

class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        # res_downconv = [ResidualBlock(inner_nc)]
        # res_upconv = [ResidualBlock(outer_nc)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv,nn.Softmax(dim=1)]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# 基本的块网络，用于堆叠形成每一个卷积块
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


# UNet++骨干网络
class NestedUNet(nn.Module):
    def __init__(self, input_channels,num_classes, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 第一斜列（左上到右下）
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # 第二斜列
        self.conv0_1 = VGGBlock(nb_filter[0] * 1 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] * 1 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] * 1 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] * 1 + nb_filter[4], nb_filter[3], nb_filter[3])

        # 第三斜列
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        # 第四斜列
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        # 第五斜列
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # 1×1卷积核
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.end=nn.Softmax(dim=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]  # 深监督，有四个损失函数共同训练

        else:
            output = self.final(x0_4)
            output=self.end(output)

            return output



'''---------------------U2Net------------------------'''

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

        self.end=nn.Softmax(dim=1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        d0=self.end(d0)

        return d0

''' ---------------------------------TPM----------------------------------------'''
class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)


class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


# cal TPS coefficient
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512,hidd_num=512,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

        )
        self.linear = nn.Linear(hidd_num, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x) #[b, 2*5*5]
        x = self.tanh(x)
        return x

class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self,opt,vgg_layer,feat_size,freeze=True):
        super().__init__()

        self.backbone =VGG19().cuda()
        self.vgg_layer =vgg_layer
        # 部分网络停止参数更新的方法
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_size=feat_size

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []
        vgg_feats=self.backbone(img)
        for l in range(len(self.vgg_layer)):
            feats.append(vgg_feats[self.vgg_layer[l]])
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear',True)
        return feats
class FlowRegression(nn.Module):
    def __init__(self, input_nc=512,output_dim=6, use_cuda=True):
        super( FlowRegression, self).__init__()

        # 线性
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, output_dim, kernel_size=3,stride=1, padding=1),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.tanh = nn.Tanh()
        self.sig=nn.Sigmoid()
        if use_cuda:
            self.conv.cuda()
            self.tanh.cuda()
            self.sig.cuda()

    def forward(self, x):
        x = self.conv(x)
        x1= self.tanh(x[:,:2,:,:])
        x2= self.sig(x[:, 2, :, :]).unsqueeze(1)
        return x1,x2

class MaskRegression(nn.Module):
    def __init__(self, input_nc=512,output_dim=6, use_cuda=True):
        super( MaskRegression, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, output_dim, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.sigm = nn.Sigmoid()
        if use_cuda:
            self.conv.cuda()
            self.sigm.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigm(x)
        return x

class TransformerAggregator(nn.Module):

    def __init__(self, opt,embed_dim,feat_size,feat_level_num,use_cuda=True):

        '''
        num_hyperpixel, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None
        '''
        super().__init__()
        norm_layer =  partial(nn.LayerNorm, eps=1e-6)
        #
        self.feat_size=feat_size

        # pos embeding
        self.pos_embed_x = nn.Parameter(
            torch.zeros(1, feat_level_num, 1,feat_size[1],embed_dim//2))  # [1,hypr_num,1, 224, 384//2]
        self.pos_embed_y = nn.Parameter(
            torch.zeros(1, feat_level_num, feat_size[0], 1,embed_dim//2))

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

        dpr = [x.item() for x in torch.linspace(0, opt.drop_path_rate, opt.tps_tfm_depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(dim=embed_dim, num_heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=opt.drop, attn_drop=opt.atten_drop, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(opt.tps_tfm_depth)])

        if use_cuda:
            self.pos_embed_x.cuda()
            self.pos_embed_y.cuda()
            self.blocks.cuda()
        #     self.blocks=self.blocks.cuda()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,  source, target):

        # [1,3,64,48,64] -> [1,3,64,48,128] -> [1,3,64*48,128]
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.feat_size[0], 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.feat_size[1], 1)),dim=4)
        pos_embed = pos_embed.flatten(2, 3).cuda()

        # [b, 3, 3072, 64] conca [b, 3, 3072, 64] -> [b, 3, 3072, 128] + [b, 3, 3072, 128]
        x = torch.cat((source, target), dim=3) + pos_embed  # [b, 3, 3072, 128*2]
        x=self.blocks(x) # [b, 3, 3072, 128*2]

        return x

''' coarse warping '''
class Coarse_Warping(nn.Module):

    def __init__(self, opt):
        super(Coarse_Warping, self).__init__()

        self.img_size = opt.img_size
        self.ngf = opt.cloth_ngf
        self.tps_scale = opt.tps_scale
        self.tps_feat_size = opt.tps_feat_size

        # vgg feature extractor
        self.extract_tps = FeatureExtractionHyperPixel(opt, opt.tps_extract_layers, opt.tps_feat_size)
        self.vgg_fea_chan_num = [64, 128, 256, 512, 512]

        # proj
        self.proj_tps = nn.ModuleList([
            nn.Linear(self.vgg_fea_chan_num[i], opt.tps_embed_dim // 2) for i in
            range(len(self.vgg_fea_chan_num))])

        # Transformer for tps
        self.warp_decoder = TransformerAggregator(opt, opt.tps_embed_dim, opt.tps_feat_size, opt.tps_level_num,use_cuda=True)  # transformer 结构

        regression_hidd_num = opt.tps_embed_dim * 4 * opt.tps_feat_size[0] * opt.tps_feat_size[1] // (8 * 8)
        self.tps_regression = FeatureRegression(input_nc=opt.tps_level_num, hidd_num=regression_hidd_num,
                                                output_dim=2 * sum([x * x for x in self.tps_scale]), use_cuda=True)
        self.gridGen = []
        for n in range(len(self.tps_scale)):
            self.gridGen.append(TpsGridGen(opt.img_size[0], opt.img_size[1], use_cuda=True, grid_size=self.tps_scale[n]))

    def forward(self, target,source,cloth_img):

        src_feats = self.extract_tps(source.clone().detach())
        tgt_feats = self.extract_tps(target.clone().detach())

        src_feats_proj = []
        tgt_feats_proj = []
        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            src_feats_proj.append(self.proj_tps[i](src.flatten(2).transpose(-1, -2)))
            tgt_feats_proj.append(self.proj_tps[i](tgt.flatten(2).transpose(-1, -2)))

        src_feats = torch.stack(src_feats_proj, dim=1)
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)

        corr = self.warp_decoder(src_feats, tgt_feats)
        theta = self.tps_regression(corr)

        tps_coeff_start,tps_coeff_num=0,0
        warp_cloth=[]
        warp_cloth_mask=[]
        grid=[]

        for x in range(len(self.tps_scale)):

            tps_coeff_start+=tps_coeff_num
            tps_coeff_num=2*self.tps_scale[x]*self.tps_scale[x]
            theta_this= theta[:,tps_coeff_start:tps_coeff_start+tps_coeff_num]
            grid_this=self.gridGen[x](theta_this)

            grid.append(grid_this)

            warped_cloth_this = F.grid_sample(cloth_img.clone().detach(), grid_this.clone(), padding_mode='border')
            warped_cloth_mask_this = F.grid_sample(source.clone().detach(), grid_this.clone(), padding_mode='border')
            warp_cloth.append(warped_cloth_this)
            warp_cloth_mask.append(warped_cloth_mask_this)

        return  warp_cloth,warp_cloth_mask,grid

''' fined mapping '''
class Fined_Mapping(nn.Module):

    def __init__(self, opt):
        super(Fined_Mapping, self).__init__()


        '''---------------------------------------------encoder-----------------------------------------------'''
        # [b,8,h,w] ->  [b,48,h,w]
        self.patch_embed_cloth = OverlapPatchEmbed(3+3, opt.flow_embed_dim)
        #  [b,48,h,w]
        self.encoder2_level1 = nn.Sequential(*[TransformerBlock(dim=opt.flow_embed_dim, num_heads=opt.flow_enc_head[0],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                                bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[0])])

        # [b,48,h,w] -> [b,48*2,h//2,w//2]
        self.enc2_down2 = Downsample(opt.flow_embed_dim)
        # [b,48*2,h//2,w//2]
        self.encoder2_level2 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim * 2 ** 1), num_heads=opt.flow_enc_head[1],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                                bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[1])])

        # [b,48*2,h//2,w//2] -> [b,48*4,h//4,w//4]
        self.enc2_down3 = Downsample(int(opt.flow_embed_dim * 2 ** 1))
        # [b,48*4,h//4,w//4]
        self.encoder2_level3 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim * 2 ** 2), num_heads=opt.flow_enc_head[2],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                                bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[2])])

        # [b,48*4,h//4,w//4] -> [b,48*8,h//8,w//8]
        self.enc2_down4 = Downsample(int(opt.flow_embed_dim * 2 ** 2))
        self.encoder2_level4 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim * 2 ** 3), num_heads=opt.flow_enc_head[3],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                                bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[3])])

        '''---------------------------------------------解码器-----------------------------------------------'''
        # [b,48*8,h//8,w//8] -> [b,48*4,h//4,w//4]
        self.up4_3 = Upsample(int(opt.flow_embed_dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(opt.flow_embed_dim * 2 ** 3), int(opt.flow_embed_dim * 2 ** 2),kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim * 2 ** 2), num_heads=opt.flow_enc_head[2],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                               bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[2])])

        #  [b,48*4,h//4,w//4] -> [b,48*2,h//2,w//2]
        self.up3_2 = Upsample(int(opt.flow_embed_dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(opt.flow_embed_dim * 2 ** 2), int(opt.flow_embed_dim * 2 ** 1),kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim * 2 ** 1), num_heads=opt.flow_enc_head[1],ffn_expansion_factor=opt.ffn_expansion_factor,
                                              bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[1])])

        #  [b,48*2,h//2,w//2] -> [b,48,h,w]
        self.up2_1 = Upsample(int(opt.flow_embed_dim * 2 ** 1))
        self.reduce_chan_level1 = nn.Conv2d(int(opt.flow_embed_dim * 2 ** 1), int(opt.flow_embed_dim), kernel_size=1,bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim), num_heads=opt.flow_enc_head[0],ffn_expansion_factor=opt.ffn_expansion_factor,
                                            bias=False, LayerNorm_type='WithBias') for i in range(opt.flow_num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(opt.flow_embed_dim), num_heads=opt.flow_enc_head[0],ffn_expansion_factor=opt.ffn_expansion_factor,
                                                           bias=False, LayerNorm_type='WithBias') for i in range(2)])
        self.end = nn.Sequential(nn.Conv2d(opt.flow_embed_dim, 3, kernel_size=1, bias=False), nn.Tanh())


    def forward(self, tgt_shape,src_img):


        inp = torch.cat((tgt_shape, src_img), dim=1)

        cloth_enc_1=self.patch_embed_cloth(inp)
        cloth_enc_1=self.encoder2_level1(cloth_enc_1) # [b,c,h,w]

        cloth_enc_2 = self.enc2_down2(cloth_enc_1)
        cloth_enc_2 = self.encoder2_level2(cloth_enc_2)  # [b,c*2,h//2,w//2]

        cloth_enc_3 = self.enc2_down3(cloth_enc_2)
        cloth_enc_3 = self.encoder2_level3(cloth_enc_3)  # [b,c*4,h//4,w//4]

        cloth_enc_4 = self.enc2_down4(cloth_enc_3)
        cloth_enc_4 = self.encoder2_level4(cloth_enc_4)  # [b,c*8,h//8,w//8]

        cloth_dec_4=self.up4_3(cloth_enc_4 ) # [b,c*4,h//4,w//4]
        cloth_dec_4_cat=torch.cat((cloth_dec_4,cloth_enc_3),dim=1)
        cloth_dec_4_reduce=self.reduce_chan_level3(cloth_dec_4_cat)
        cloth_dec_4_end=self.decoder_level3(cloth_dec_4_reduce)

        cloth_dec_3=self.up3_2(cloth_dec_4_end) # [b,c*2,h//2,w//2]
        cloth_dec_3_cat=torch.cat((cloth_dec_3,cloth_enc_2),dim=1)
        cloth_dec_3_reduce=self.reduce_chan_level2(cloth_dec_3_cat)
        cloth_dec_3_end=self.decoder_level2(cloth_dec_3_reduce)

        cloth_dec_2=self.up2_1(cloth_dec_3_end) # [b,c,h,w]
        cloth_dec_2_cat = torch.cat((cloth_dec_2, cloth_enc_1), dim=1)
        cloth_dec_2_reduce = self.reduce_chan_level1(cloth_dec_2_cat)
        cloth_dec_2_end=self.decoder_level1(cloth_dec_2_reduce)

        cloth_dec_1=self.refinement(cloth_dec_2_end) # [b,c,h,w]
        synthesis_img=self.end(cloth_dec_1)

        return synthesis_img


''' composition '''
class Composition(nn.Module):

    def __init__(self, opt):
        super(Composition, self).__init__()

        self.composition_ngf = opt.composition_ngf
        self.composition_g = UnetGenerator(6, 3, 4, ngf=self.composition_ngf, norm_layer=nn.InstanceNorm2d)
        self.composition_mask_norm = nn.Sigmoid()

    def forward(self, src_img, syn_img):

        inp = torch.cat((src_img.clone().detach(), syn_img.clone().detach()), dim=1)
        composition_mask = self.composition_g(inp)
        composition_mask_end = self.composition_mask_norm(composition_mask)

        # 生成 img 和 composition mask
        composition_img =  src_img * composition_mask_end + syn_img* (1 - composition_mask_end)

        return composition_mask_end, composition_img


class ViT(nn.Module):

    def __init__(self, opt):
        super(ViT, self).__init__()

        self.tps_scale =len(opt.tps_scale)
        self.img_size=opt.img_size

        self.coarse_warping=Coarse_Warping(opt)
        self.fined_mapping=Fined_Mapping(opt)
        self.composition=Composition(opt)

    def forward(self, tgt_mask,src_mask,src_img):

        self.batch=tgt_mask.shape[0]

        # coarse stage
        coarse_img_set,coarse_mask_set,grid_set=self.coarse_warping(tgt_mask.clone().detach(),src_mask.clone().detach(),src_img)

        # find best coarse result
        shape_mse_set =torch.FloatTensor(size=(self.batch,self.tps_scale)).cuda()
        for n in range(len(coarse_mask_set)):
            shape_mse = 0.2*torch.abs(coarse_mask_set[n].clone().detach()-tgt_mask.clone().detach())+0.8*torch.abs(coarse_mask_set[n].clone().detach()-src_mask.clone().detach())
            shape_mse_value = torch.mean(shape_mse, dim=(1, 2, 3))
            shape_mse_set[:,n]=shape_mse_value
        index=torch.argmin(shape_mse_set,dim=1)

        coarse_img = torch.FloatTensor(size=(self.batch, 3,self.img_size[0],self.img_size[1])).cuda()
        coarse_mask = torch.FloatTensor(size=(self.batch, 3, self.img_size[0], self.img_size[1])).cuda()
        for n in range(self.batch):
            coarse_img[n,:,:,:]=(coarse_img_set[index[n]][n,:,:,:]).clone().detach()
            coarse_mask[n, :, :, :] = (coarse_mask_set[index[n]][n, :, :, :]).clone().detach()

        # fined mapping stage
        fined_img=self.fined_mapping(tgt_mask.clone().detach(),coarse_img.clone().detach())
        coarse_img_fake = self.fined_mapping(coarse_mask.clone().detach(),fined_img)

        # composition stage
        composition_mask, composition_img=self.composition(coarse_img.clone().detach(),fined_img.clone().detach())

        return coarse_img_set+[coarse_img,fined_img,coarse_img_fake,composition_img],coarse_mask_set+[coarse_mask,composition_mask],index,grid_set


'''------------------------RSIM---------------------------'''
class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

        self.end=nn.Tanh()

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        x=self.end(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle):
        # x, rgb, style
        if exists(self.upsample):
            x = self.upsample(x)

        # print(istyle.shape)
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size[0]) - 4) # 8-3

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1] # [32,64,128,256,512,512,512,512]

        set_fmap_max = partial(min, fmap_max) # 截断512 # [32,64,128,256,512,512,512,512]
        filters = list(map(set_fmap_max, filters)) # [32,64,128,256,512,512,512,512]
        init_channels = filters[0]
        filters = [init_channels, *filters] # [32,32,64,128,256,512,512,512,512]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, shap):

        x=shap
        rgb = None
        x = self.initial_conv(x)

        for  block, attn in zip( self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, styles)



        return rgb

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class VGG_Pre(nn.Module):
    def __init__(self,opt,freeze=True):
        super().__init__()

        self.backbone =VGG19().cuda()
        self.vgg_layer =opt.vgg_layer
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []
        vgg_feats=self.backbone(img)
        for l in range(len(self.vgg_layer)):
            feats.append(vgg_feats[self.vgg_layer[l]])
        return feats

class Arm_Pooling(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.channel_reduce=[]
        self.channel_num=[128,256,512]
        self.vgg_layer=opt.vgg_layer
        self.pool_size=opt.pool_size
        self.reduce_chan=opt.reduce_chan

        self.channel_reduce_1=nn.Conv2d(self.channel_num[0], opt.reduce_chan, kernel_size=1, bias=False).cuda()
        self.channel_reduce_2 = nn.Conv2d(self.channel_num[1], opt.reduce_chan, kernel_size=1, bias=False).cuda()
        self.channel_reduce_3= nn.Conv2d(self.channel_num[2], opt.reduce_chan, kernel_size=1, bias=False).cuda()

        self.pooling_layer_0 = nn.AdaptiveAvgPool2d(self.pool_size).cuda()
        self.pooling_layer_1=nn.AdaptiveAvgPool2d(self.pool_size).cuda()
        self.pooling_layer_2 = nn.AdaptiveAvgPool2d(self.pool_size).cuda()
        self.pooling_layer_3 = nn.AdaptiveAvgPool2d(self.pool_size).cuda()


    def forward(self,vgg_fea):

        [b,_,_,_]=vgg_fea[0].shape
        vgg_fea_pooling=torch.empty(size=(b,len(self.vgg_layer),self.reduce_chan,self.pool_size[0],self.pool_size[1])).cuda()


        vgg_fea_pooling[:,0,:,:,:]=self.pooling_layer_0(vgg_fea[0])
        vgg_fea_pooling[:,1, :, :, :] = self.channel_reduce_1(self.pooling_layer_1(vgg_fea[1]))
        vgg_fea_pooling[:, 2, :, :, :] = self.channel_reduce_2(self.pooling_layer_2(vgg_fea[2]))
        vgg_fea_pooling[:, 3, :, :, :] = self.channel_reduce_3(self.pooling_layer_3(vgg_fea[3]))

        vgg_fea_pooling=vgg_fea_pooling.view(b,-1)

        return vgg_fea_pooling

class arm_paint_model(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.map= StyleVectorizer(emb=opt.reduce_chan*opt.pool_size[0]*opt.pool_size[1]*len(opt.vgg_layer), depth=8)
        self.g= Generator(image_size=opt.img_size,
                          latent_dim=opt.reduce_chan*opt.pool_size[0]*opt.pool_size[1]*len(opt.vgg_layer),
                          network_capacity=32,
                          transparent = False,
                          attn_layers=[],
                          no_const=False,
                          fmap_max=512
                          )
        self.arm_vgg=VGG_Pre(opt)
        self.arm_pool=Arm_Pooling(opt)

    def forward(self,human_img,human_mask,human_mask_eras,human_mask_diff):

        arm=human_img*human_mask_eras[:,3,:,:].unsqueeze(1)
        arm_vgg=self.arm_vgg(arm)
        arm_vgg_pool=self.arm_pool(arm_vgg)
        arm_w=self.map(arm_vgg_pool)

        arm_shape_vgg=self.arm_vgg(human_mask[:,3,:,:].unsqueeze(1).repeat(1,3,1,1))[-1] # 512*4*3

        arm_pred=self.g(arm_w,arm_shape_vgg)
        arm_comp=arm_pred*human_mask_diff[:,3,:,:].unsqueeze(1)+human_img*human_mask_eras[:,3,:,:].unsqueeze(1)

        return arm_pred,arm_comp