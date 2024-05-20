from network import *
import itertools
from skimage.draw import ellipse as circle
from skimage.draw import  line_aa, polygon
import torch
import os
import numpy as np
import cv2 as cv
import datetime
import time
import matplotlib.pyplot as plt
import csv
from torchvision.utils import save_image

class viton_model():

    def __init__(self,opt):

        super(viton_model,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device
        self.ngf=opt.ngf

        # generator
        self.g = ViT(opt)

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # eval
        self.data_dir=opt.data_dir
        self.checkpoint_dir = opt.checkpoint_dir
        self.eval_dir=opt.eval_dir

        # load checkpoint
        self.inference=opt.inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir,"%PTM_d_G.pth"%self.inference), map_location=torch.device(self.device)))

        # optimizer
        self.lr=opt.g_lr
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=opt.g_lr, betas=(0.5, 0.999))

        # loss
        self.l1_func=torch.nn.L1Loss()
        self.l2_func = torch.nn.MSELoss()
        self.vgg_func = VGG19().cuda()
        self.vgg_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

        # hyper para
        self.coarse_loss_coeff=opt.coarse_loss_coeff
        self.fined_l1_loss_coeff = opt.fined_l1_loss_coeff
        self.fined_vgg_loss_coeff = opt.fined_vgg_loss_coeff
        self.composition_loss_coeff=opt.composition_loss_coeff
        self.grid_loss_coeff=opt.grid_loss_coeff
        self.coarse_result_weight = opt.coarse_result_weight
        self.grouth_truth_weight = opt.grouth_truth_weight

        # warp
        self.warp=bilinear_warp

        # vis
        self.vis_dir=opt.vis_dir
        self.vis_size=opt.img_size
        self.batch=opt.batch
        self.human_sg_num = opt.human_sg_num

        # log
        self.epoch=opt.epoch
        self.ite_num=int(len(os.listdir(os.path.join(opt.data_dir,opt.cloth_dir)))/opt.batch)
        self.loss_dir=opt.loss_dir
        self.loss_list=[[] for i in range(10)]
        self.index_list=[]

    def setinput(self,input):

        self.input=input

        cloth_img, cloth_mask, human_img, human_mask=self.input[0],self.input[1],self.input[2],self.input[3]

        if len(self.gpu_ids) > 0:
            self.cloth_img=cloth_img.cuda()
            self.cloth_mask=cloth_mask.cuda()
            self.human_img=human_img.cuda()
            self.human_mask=human_mask.cuda()

    def vgg_cal(self, x, y, style=False):

        x_vgg = self.vgg_func(x)
        y_vgg = self.vgg_func(y)

        loss = 0.0
        for l in range(len(self.vgg_layer)):
            loss += self.l1_func(x_vgg[self.vgg_layer[l]], y_vgg[self.vgg_layer[l]])
        return loss

    def forward(self):

        ''' forward '''
        cloth_on_body_mask=self.human_mask[:,4,:,:].unsqueeze(1).repeat(1,3,1,1)
        cloth_in_shop_mask = self.cloth_mask.repeat(1, 3, 1, 1)
        cloth_in_shop=self.cloth_mask*self.cloth_img
        cloth_on_body = cloth_on_body_mask* self.human_img

        (b,_,_,_)=cloth_in_shop_mask.shape
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, 192), np.linspace(-1, 1, 256))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3).cuda().repeat(b,1,1,1)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3).cuda().repeat(b,1,1,1)
        grid_source = torch.cat((grid_X, grid_Y), 3)

        self.warped_cloth,self.warped_mask,_,self.warped_grid=self.g(cloth_on_body_mask,cloth_in_shop_mask,cloth_in_shop)

        ''' coarse loss'''
        self.coarse_loss=0.0
        for n in range(len(self.warped_mask)-2):
            self.coarse_loss += self.l1_func(self.warped_mask[n],cloth_on_body_mask)

        ''' grid loss '''
        self.grid_loss=0.0
        for n in range(len(self.warped_grid)):
            self.grid_loss+=self.l2_func(grid_source.permute(0,3,1,2),self.warped_grid[n].permute(0,3,1,2))

        ''' fined l1 loss'''
        self.fined_l1_loss=0.0
        self.fined_l1_loss += self.l1_func(self.warped_cloth[4], self.warped_cloth[6])
        self.fined_l1_loss += self.l1_func(self.warped_cloth[5],self.warped_cloth[4])*self.coarse_result_weight
        self.fined_l1_loss += self.l1_func(self.warped_cloth[5],cloth_on_body)*self.grouth_truth_weight

        ''' fined vgg loss'''
        self.fined_vgg_loss = 0.0
        self.fined_vgg_loss += self.vgg_cal(self.warped_cloth[5], self.warped_cloth[4]) * self.coarse_result_weight
        self.fined_vgg_loss += self.vgg_cal(self.warped_cloth[5], cloth_on_body) * self.grouth_truth_weight

        inner_mask = cloth_on_body_mask*self.warped_mask[-2]
        self.composition_loss=0.0
        self.composition_loss+=self.l1_func(self.warped_cloth[-1], cloth_on_body)*self.grouth_truth_weight
        self.composition_loss += self.l1_func(self.warped_cloth[-1]* inner_mask, self.warped_cloth[4]*inner_mask)*self.coarse_result_weight

        ''' loss '''
        self.loss=self.coarse_loss*self.coarse_loss_coeff+self.fined_l1_loss*self.fined_l1_loss_coeff+self.fined_vgg_loss*self.fined_vgg_loss_coeff\
                  +self.composition_loss*self.composition_loss_coeff+self.grid_loss*self.grid_loss_coeff

        ''' backward '''
        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def vis_result(self,epo,ite):

        for m in range(self.batch):
            vis_path = os.path.join(self.vis_dir, 'E{}_I{}_B{}.jpg'.format(epo, ite, m))
            human_img = self.de_norm(Variable(self.human_img[m]).data)
            cloth_img = self.de_norm(Variable(self.cloth_img[m]).data)

            warped_cloth_img = []
            for k in range(len(self.warped_cloth)):
                warped_cloth_img.append(self.de_norm(Variable(self.warped_cloth[k][m]).data))
            for k in range(len(self.warped_mask)):
                warped_cloth_img.append(self.de_norm(Variable(self.warped_mask[k][m]).data))

            other_mask = (self.human_mask[m, 0, :, :] + self.human_mask[m, 1, :, :] + self.human_mask[m, 2, :,:] + self.human_mask[m, 3, :,:]).unsqueeze(0)
            composition = human_img * other_mask + warped_cloth_img[7] * self.human_mask[m, 4, :, :].unsqueeze(0)
            warped_cloth_img.append(composition)

            vis_all = torch.cat([human_img, cloth_img] + [x for x in warped_cloth_img], dim=-1)
            save_image(vis_all, vis_path, normalize=True)

    def log_print(self,epo,ite,time_bench):

        # 耗时
        elapsed = time .time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo+1,self.epoch, ite + 1, self.ite_num)

        # loss
        log += ",g_Loss: {:.4f}".format(self.loss.item())
        log += ",coarse_Loss: {:.4f}".format(self.coarse_loss.item()*self.coarse_loss_coeff)
        log += ",fined_L1_Loss: {:.4f}".format(self.fined_l1_loss.item() * self.fined_l1_loss_coeff)
        log += ",fined_VGG_Loss: {:.4f}".format(self.fined_vgg_loss.item() * self.fined_vgg_loss_coeff)
        log += ",composition_Loss: {:.4f}".format(self.composition_loss.item()*self.composition_loss_coeff)
        log += ",grid_Loss: {:.4f}".format(self.grid_loss.item()*self.grid_loss_coeff)

        print(log)

    def plot_loss(self,epo,ite):

        ite_sum=epo* self.ite_num+ite
        self.index_list.append(ite_sum)

        loss_list=[self.loss.item(),
                   self.coarse_loss.item()*self.coarse_loss_coeff,
                   self.fined_l1_loss.item()*self.fined_l1_loss_coeff,
                   self.fined_vgg_loss.item() * self.fined_vgg_loss_coeff,
                   self.composition_loss.item()*self.composition_loss_coeff,
                   self.grid_loss.item()*self.grid_loss_coeff]
        loss_name=["g_loss","coarse_loss","fined_l1_loss","fined_vgg_loss","composition_Loss","grid_Loss"]

        for m in range(len(loss_list)):

            self.loss_list[m].append(loss_list[m])
            plt.figure()
            plt.plot(self.index_list,self.loss_list[m], 'b', label=loss_name[m])
            plt.ylabel(loss_name[m])
            plt.xlabel('iter_num')
            plt.legend()
            plt.savefig(os.path.join(self.loss_dir, "{}.jpg".format(loss_name[m])))
            plt.cla()
            plt.close("all")

    def save_network(self,epo):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, 'PTM_{}_G.pth'.format(epo+1)))

    def print_network(self):

        model=[self.g]

        num_params = 0
        for k in range(len(model)):
            for p in  model[k].parameters():
                num_params += p.numel()
        for k in range(len(model)):
            print(model[k])
        print("The number of parameters: {}".format(num_params))

    def eval(self, ite):
        cloth_on_body_mask=self.human_mask[:,4,:,:].unsqueeze(1).repeat(1,3,1,1)
        cloth_in_shop_mask=self.cloth_mask.repeat(1,3,1,1)
        warped_cloth,_ ,_,_ = self.g(cloth_on_body_mask, cloth_in_shop_mask,self.cloth_img*cloth_in_shop_mask)
        vis_path = os.path.join(self.eval_dir, '%06d.jpg'%ite)
        warped_cloth_img = self.de_norm(Variable(warped_cloth[-1]).data)*cloth_on_body_mask+(1-cloth_on_body_mask)*torch.ones(size=(3,256,192)).cuda()
        save_image(warped_cloth_img, vis_path, normalize=True)



