from network import *
import itertools
from skimage.draw import circle, line_aa, polygon
import torch
import os
import numpy as np
import cv2 as cv
import datetime
import time
import matplotlib.pyplot as plt
import csv
from torchvision.utils import save_image

class armpaint_model():

    def __init__(self,opt):

        super(armpaint_model,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device

        # generator
        self.g = arm_paint_model(opt)

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # eval
        self.checkpoint_dir = opt.checkpoint_dir
        self.eval_dir = opt.eval_dir

        # load checkpoint
        self.inference=opt.inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir,"RSIM_%d_G.pth"%self.inference), map_location=torch.device(self.device)))

        self.lr=opt.g_lr
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=opt.g_lr, betas=(0.5, 0.999))

        self.l1_func=torch.nn.L1Loss()
        self.l2_func = torch.nn.MSELoss()
        self.vgg_func = VGG19().cuda()
        self.vgg_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

        # hyper para
        self.l1_loss_coeff=opt.rsim_l1_loss_coeff
        self.l2_loss_coeff = opt.l2_loss_coeff
        self.content_loss_coeff=opt.content_loss_coeff

        # vis
        self.vis_dir=opt.vis_dir
        self.vis_size=opt.img_size
        self.batch=opt.batch
        self.human_sg_num = opt.human_sg_num

        # log
        self.epoch=opt.epoch
        self.ite_num=int(len(os.listdir(os.path.join(opt.data_dir,opt.human_dir)))/opt.batch)
        self.loss_dir=opt.loss_dir
        self.loss_list=[[] for i in range(10)]
        self.index_list=[]

    def setinput(self,input):

        self.input=input

        human_img, human_mask, human_mask_eras,human_mask_diff=self.input[0],self.input[1],self.input[2],self.input[3]

        if len(self.gpu_ids) > 0:
            self.human_img=human_img.cuda()
            self.human_mask=human_mask.cuda()
            self.human_mask_eras= human_mask_eras.cuda()
            self.human_mask_diff=human_mask_diff.cuda()

    def vgg_cal(self, x, y, style=False):

        x_vgg = self.vgg_func(x)
        y_vgg = self.vgg_func(y)

        loss = 0.0
        for l in range(len(self.vgg_layer)):
            loss += self.l1_func(x_vgg[self.vgg_layer[l]], y_vgg[self.vgg_layer[l]])
        return loss

    def forward(self):

        ''' forward '''
        self.arm=self.human_img*self.human_mask[:,3,:,:].unsqueeze(1)
        self.arm_pred,self.arm_comp=self.g(self.human_img,self.human_mask,self.human_mask_eras,self.human_mask_diff)

        ''' L1 loss'''
        self.l1_loss=self.l1_func(self.arm_pred,self.arm)
        self.l1_loss+= self.l1_func(self.arm_comp, self.arm)

        self.l2_loss = self.l2_func(self.arm_pred,self.arm)
        self.l2_loss += self.l2_func(self.arm_comp, self.arm)

        self.content_loss=self.vgg_cal(self.arm_pred,self.arm)
        self.content_loss+= self.vgg_cal(self.arm_comp, self.arm)

        ''' loss function '''
        self.loss=self.l1_loss*self.l1_loss_coeff+self.l2_loss*self.l2_loss_coeff+ self.content_loss*self.content_loss_coeff

        ''' optimizer '''
        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def vis_result(self,epo,ite):

        self.arm_input = self.human_img * self.human_mask_eras[:, 3, :, :].unsqueeze(1)

        for m in range(self.batch):
            vis_path=os.path.join(self.vis_dir, 'E{}_I{}_B{}.jpg'.format(epo, ite, m))
            arm_input=self.de_norm(Variable( self.arm_input[m]).data)
            arm_label=self.de_norm(Variable(self.arm[m]).data)
            arm_pred= self.de_norm(Variable(self.arm_pred[m]).data)
            arm_comp = self.de_norm(Variable(self.arm_comp[m]).data)

            vis_all=torch.cat([arm_input,arm_label,arm_pred,arm_comp],dim=-1)
            save_image(vis_all, vis_path, normalize=True)

    def log_print(self,epo,ite,time_bench):

        # time
        elapsed = time .time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo+1,self.epoch, ite + 1, self.ite_num)

        # loss
        log += ",g_Loss: {:.4f}".format(self.loss.item())
        log += ",l1_Loss: {:.4f}".format(self.l1_loss.item()*self.l1_loss_coeff)
        log += ",l2_Loss: {:.4f}".format(self.l2_loss.item() * self.l2_loss_coeff)
        log += ",cont_Loss: {:.4f}".format(self.content_loss.item() * self.content_loss_coeff)

        print(log)

    def plot_loss(self,epo,ite):

        ite_sum=epo* self.ite_num+ite
        self.index_list.append(ite_sum)

        loss_list=[self.loss.item(),self.l1_loss.item()*self.l1_loss_coeff,self.l2_loss.item()*self.l2_loss_coeff,self.content_loss.item() * self.content_loss_coeff]
        loss_name=["g_loss","l1_loss","l2_loss","cont_loss"]

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

        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, 'RSIM_{}_G.pth'.format(epo+1)))

    def print_network(self):

        model=[self.g]

        num_params = 0
        for k in range(len(model)):
            for p in  model[k].parameters():
                num_params += p.numel()
        for k in range(len(model)):
            print(model[k])
        print("The number of parameters: {}".format(num_params))

    # test
    def eval(self, ite):

        ''' 预测 '''
        arm_pred,arm_comp= self.g(self.human_img,self.human_mask,self.human_mask_eras,self.human_mask_diff)
        vis_path = os.path.join(self.eval_dir, '%06d.jpg'%ite)
        arm_comp= self.de_norm(Variable(arm_comp).data)
        arm_pred = self.de_norm(Variable(arm_pred).data)
        save_image( arm_comp, vis_path, normalize=True)



