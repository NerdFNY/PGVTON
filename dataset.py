import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image
import random
import math

class pgvton_dataset(data.Dataset):

    def __init__(self):
        super(pgvton_dataset,self).__init__()

    def initialize(self,opt):

        # dir
        self.cloth_mask_dir=os.path.join(opt.data_dir,opt.cloth_mask_dir)
        self.human_mask_dir=os.path.join(opt.data_dir,opt.human_mask_dir)
        self.densepose_dir = os.path.join(opt.data_dir, opt.densepose_dir)
        self.openpose_dir=os.path.join(opt.data_dir,opt.openpose_dir)

        # files
        self.cloth_mask_files=os.listdir(self.cloth_mask_dir)
        self.human_mask_files=os.listdir(self.human_mask_dir)
        self.densepose_files=os.listdir(self.densepose_dir)

        # num
        self.human_sg_num=opt.human_sg_num
        self.data_size=len(self.human_mask_files)
        self.img_size=opt.img_size
        self.joint_num=opt.joint_num
        self.miss_value=-1
        self.sigma=opt.sigma

        # process
        transform_list = []
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

        self.aug=transforms.Compose([transforms.RandomAffine(degrees=8,translate=(0.075, 0.075),scale=(0.9,1.02))])

    def __getitem__(self,index):

        ''' path '''
        cloth_mask_path = os.path.join(self.cloth_mask_dir, self.cloth_mask_files[index])
        index_random=random.randint(0, self.data_size-1)
        cloth_mask_rand_path = os.path.join(self.cloth_mask_dir, self.cloth_mask_files[index_random])
        human_mask_path=os.path.join(self.human_mask_dir,self.human_mask_files[index])
        densepose_path=os.path.join(self.densepose_dir,self.densepose_files[index])

        '''human_mask'''
        sg_img = Image.open(human_mask_path)
        sg_img = np.expand_dims(np.array(sg_img)[:, :, 0], 0)

        sg_img_1d = torch.from_numpy(sg_img).view(-1).long()
        ones = torch.sparse.torch.eye(self.human_sg_num)  # onehot
        ones = ones.index_select(0, sg_img_1d)
        sg_onehot = ones.view([self.img_size[0], self.img_size[1], self.human_sg_num])
        human_mask= sg_onehot.permute(2, 0, 1)
        human_mask_label=torch.from_numpy(sg_img).long()


        '''cloth_mask'''
        cloth_mask = Image.open(cloth_mask_path)
        cloth_mask = np.round(np.array(cloth_mask) / 255)
        cloth_mask = np.expand_dims(cloth_mask, 0)
        cloth_mask = torch.from_numpy(cloth_mask).long()

        cloth_mask_rand= Image.open(cloth_mask_rand_path)
        cloth_mask_rand = np.round(np.array(cloth_mask_rand) / 255)
        cloth_mask_rand = np.expand_dims(cloth_mask_rand, 0)
        cloth_mask_rand = torch.from_numpy(cloth_mask_rand).long()

        '''densepose'''
        densepose=np.load(densepose_path)
        densepose = torch.from_numpy(densepose).long().float()


        return human_mask,cloth_mask,cloth_mask_rand,densepose,human_mask_label

    def __len__(self):
        return  self.data_size

    def joint2map(self,J):

        joint=np.zeros(shape=(2,self.joint_num))
        joint[0,:]=np.array(J[0][:])
        joint[1, :] = np.array(J[1][:])

        joint=joint.astype(float)
        map=np.zeros(shape=(self.img_size[0],self.img_size[1],self.joint_num),dtype='float32')

        for i in range(joint.shape[1]):

            if joint[0,i]==self.miss_value or joint[1,i]==self.miss_value:
                continue

            joint_x=int(joint[0,i])
            joint_y=int(joint[1,i])

            xx,yy = np.meshgrid(np.arange(self.img_size[1]), np.arange(self.img_size[0]))

            map[:,:,i]=np.exp(-((yy - joint_y) ** 2 + (xx - joint_x) ** 2) / (2 * self.sigma ** 2))

        return map


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.擦除面积与输入图像的最小比例
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.擦除面积的最小纵横比
         mean: Erasing value.
    """
    def __init__(self,   probability=0.1, sl=0.02, sh=0.50, r1=0.3, mean=0.0):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        for attempt in range(20):

            if random.uniform(0, 1) >= self.probability :

                area = img.size()[1] * img.size()[2]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    if img.size()[0] == 7:
                        img[3, x1:x1 + h, y1:y1 + w] = self.mean
                    else:
                        img[3, x1:x1 + h, y1:y1 + w] = self.mean

        return img


class armpaint_dataset(data.Dataset):

    def __init__(self):
        super(armpaint_dataset,self).__init__()

    def initialize(self,opt):

        # dir
        self.human_img_dir=os.path.join(opt.data_dir,opt.human_dir)
        self.human_mask_dir=os.path.join(opt.data_dir,opt.human_mask_dir)
        self.human_mask_src_dir = os.path.join(opt.data_dir, opt.human_mask_src_dir)
        self.mode=opt.mode

        # files
        self.human_img_files=os.listdir(self.human_img_dir)
        self.human_mask_files=os.listdir(self.human_mask_dir)
        self.human_mask_src_files = os.listdir(self.human_mask_src_dir)

        # num
        self.human_sg_num=opt.human_sg_num
        self.data_size=len(self.human_img_files)
        self.img_size=opt.img_size

        # process
        transform_list = []
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)
        self.aug = transforms.Compose([RandomErasing()])

    def __getitem__(self,index):

        ''' path '''
        human_img_path=os.path.join(self.human_img_dir,self.human_img_files[index])
        human_mask_path=os.path.join(self.human_mask_dir,self.human_mask_files[index])
        human_mask_src_path = os.path.join(self.human_mask_src_dir, self.human_mask_src_files[index])

        human_img=Image.open(human_img_path)
        human_img=self.trans(human_img)

        sg_img = Image.open(human_mask_path)
        sg_img = np.expand_dims(np.array(sg_img)[:, :, 0], 0)

        sg_img_1d = torch.from_numpy(sg_img).view(-1).long()
        ones = torch.sparse.torch.eye(self.human_sg_num)  # onehot
        ones = ones.index_select(0, sg_img_1d)
        sg_onehot = ones.view([self.img_size[0], self.img_size[1], self.human_sg_num])
        human_mask= sg_onehot.permute(2, 0, 1)

        sg_img_src = Image.open(human_mask_src_path)
        sg_img_src = np.expand_dims(np.array(sg_img_src)[:, :, 0], 0)

        sg_img_1d_src = torch.from_numpy(sg_img_src).view(-1).long()
        ones_src = torch.sparse.torch.eye(self.human_sg_num)  # onehot
        ones_src = ones_src.index_select(0, sg_img_1d_src)
        sg_onehot_src = ones_src.view([self.img_size[0], self.img_size[1], self.human_sg_num])
        human_mask_src = sg_onehot_src.permute(2, 0, 1)

        if self.mode=="train":
            human_mask_eras=self.aug(human_mask.clone())
        else:
            human_mask_eras = human_mask_src
        human_mask_diff=human_mask.clone()-human_mask_eras.clone()


        return human_img,human_mask,human_mask_eras,human_mask_diff

    def __len__(self):
        return  self.data_size
