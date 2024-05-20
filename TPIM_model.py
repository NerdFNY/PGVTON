
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

class sginfer_model():

    def __init__(self,opt):

        super(sginfer_model,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device
        self.img_size=opt.img_size

        self.g=U2NET(4+3+1,opt.human_sg_num)

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # eval
        self.checkpoint_dir = opt.checkpoint_dir
        self.eval_dir = os.path.join(opt.data_dir,opt.eval_dir)

        # load checkpoint
        self.inference=opt.inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir,"TPIM_%d_G.pth"%self.inference), map_location=torch.device(self.device)))

        # optimizer
        self.lr=opt.g_lr
        self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=opt.g_lr, betas=(0.5, 0.999))

        # loss function
        self.l1_func=torch.nn.L1Loss()
        self.ce_func = torch.nn.CrossEntropyLoss()

        # hyper-parameter
        self.l1_loss_coeff=opt.l1_loss_coeff
        self.ce_loss_coeff=opt.ce_loss_coeff

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

        human_mask,cloth_mask,cloth_mask_rand,densepose,human_mask_label=self.input[0],self.input[1],self.input[2],self.input[3],self.input[4]

        if len(self.gpu_ids) > 0:
            self.human_mask=human_mask.cuda()
            self.cloth_mask=cloth_mask.cuda()
            self.cloth_mask_rand=cloth_mask_rand.cuda()
            self.densepose=densepose.cuda()
            self.human_mask_label=human_mask_label.cuda()

    def forward(self):

        ''' forward '''
        self.l1_loss=0.0

        # person A + garment B
        input1=torch.cat((self.human_mask[:,[1,2,5,6],:,:],self.cloth_mask_rand,self.densepose),dim=1)
        self.pred_human_mask1=self.g(input1)
        self.l1_loss+=self.l1_func(self.human_mask[:,[1,2,5,6],:,:],self.pred_human_mask1[:,[1,2,5,6],:,:])

        # person A + garment A
        input2 = torch.cat((self.pred_human_mask1[:, [1, 2, 5, 6], :, :], self.cloth_mask, self.densepose), dim=1)
        self.pred_human_mask2 = self.g(input2)

        self.l1_loss+=self.l1_func(self.pred_human_mask2,self.human_mask)
        self.ce_loss=self.ce_func(self.pred_human_mask2,self.human_mask_label.squeeze(1).long())
        self.loss=self.l1_loss*self.l1_loss_coeff+self.ce_loss*self.ce_loss_coeff

        ''' backward'''
        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def par2img(self,par_map,vis=True):

        if vis:
            vis_color=torch.cuda.FloatTensor([(0,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,255),(0,255,0),
                                              (255,0,0),(244,164,96),(255,255,255),(160,32,240)]).long()
        else:
            vis_color = torch.cuda.FloatTensor([(0,0,0), (1,1,1), (2,2, ), (3,3,3), (4,4,4),
                                                (5,5,5), (6,6,6),(7,7,7), (8,8,8), (9,9,9)]).long()
        index=torch.argmax(par_map,dim=-1).view(-1)
        ones = vis_color.index_select(0, index)
        par=ones.view(self.img_size[0],self.img_size[1],3).long()

        return par

    def vis_result(self,epo,ite):

        for m in range(self.batch):
            vis_path = os.path.join(self.vis_dir, 'E{}_I{}_B{}.jpg'.format(epo, ite, m))
            human_mask_end = self.par2img(self.human_mask[m].permute(1, 2, 0)).clone().detach()
            pred_human_mask1_end=self.par2img(self.pred_human_mask1[m].permute(1,2,0)).clone().detach()
            pred_human_mask2_end = self.par2img(self.pred_human_mask2[m].permute(1, 2, 0)).clone().detach()

            output=torch.cat((human_mask_end,self.cloth_mask_rand[m].permute(1,2,0).repeat(1,1,3)*255,pred_human_mask1_end,
                              self.cloth_mask[m].permute(1,2,0).repeat(1,1,3)*255,pred_human_mask2_end,),dim=1)
            cv.imwrite(vis_path, output.cpu().numpy())

    def log_print(self,epo,ite,time_bench):

        elapsed = time .time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo+1,self.epoch, ite + 1, self.ite_num)

        # loss
        log += ",g_Loss: {:.4f}".format(self.loss.item())
        log += ",l1_Loss: {:.4f}".format(self.l1_loss.item()*self.l1_loss_coeff)
        log += ",ce_Loss: {:.4f}".format(self.ce_loss.item() * self.ce_loss_coeff)
        print(log)

    def plot_loss(self,epo,ite):

        ite_sum=epo* self.ite_num+ite
        self.index_list.append(ite_sum)

        loss_list=[self.loss.item(),self.l1_loss.item()*self.l1_loss_coeff,self.ce_loss.item()*self.ce_loss_coeff]
        loss_name=["g_loss","l1_loss","ce_loss"]

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
        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, 'TPIM_{}_G.pth'.format(epo+1)))

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

        ''' 预测 '''
        self.human_mask_remove = self.human_mask[:, [ 1, 2, 5, 6], :, :]

        input = torch.cat((self.human_mask_remove, self.cloth_mask, self.densepose), dim=1)
        self.pred_human_mask = self.g(input)

        human_mask_end = self.par2img(self.human_mask[0].permute(1, 2, 0)).clone()
        human_mask_remove_end = self.par2img(self.human_mask_remove[0].permute(1, 2, 0)).clone()
        pred_human_mask_end = self.par2img(self.pred_human_mask[0].permute(1, 2, 0)).clone()

        output = torch.cat((self.cloth_mask[0].permute(1, 2, 0).repeat(1, 1, 3) * 255, human_mask_remove_end,human_mask_end, pred_human_mask_end), dim=1)
        vis_path_vis = os.path.join(self.eval_dir + "_vis", '%06d.jpg' % ite)
        cv.imwrite(vis_path_vis, output.cpu().numpy())

        pred_human_mask_sg = self.par2img(self.pred_human_mask[0].permute(1, 2, 0),vis=False).clone()
        vis_path_sg = os.path.join(self.eval_dir + "_sg", '%06d.png' % ite)
        cv.imwrite(vis_path_sg, pred_human_mask_sg.cpu().numpy())

