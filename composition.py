import numpy as np
import os
import cv2 as cv
from skimage import io
from skimage.measure import *
import skimage
from tqdm import tqdm

def composition():

    pred_sg_dir='your_dir\pred_sg\sg'
    hum_img_dir='your_dir\image'
    warp_gar_dir='your_dir\warped_garment'
    arm_dir='your_dir\\arm'
    output_dir="your_dir\\composition"

    pred_sg_file=os.listdir(pred_sg_dir)
    hum_img_file=os.listdir(hum_img_dir)
    warp_gar_file=os.listdir(warp_gar_dir)
    arm_dir_file=os.listdir(arm_dir)

    for i in range(len(pred_sg_file)):

        print(i)

        pred_sg_path = os.path.join(pred_sg_dir,pred_sg_file[i])
        hum_img_path =os.path.join(hum_img_dir, hum_img_file[i])
        warp_gar_path = os.path.join(warp_gar_dir,warp_gar_file[i])
        arm_path = os.path.join(arm_dir,arm_dir_file[i])
        output_path = os.path.join(output_dir, "%06d.png"%i)

        pred_sg=cv.imread(pred_sg_path)
        hum_img = cv.imread(hum_img_path)
        warp_gar = cv.imread(warp_gar_path)
        arm = cv.imread(arm_path)

        # rest part
        X1_1, Y1_1 = np.where(pred_sg[:, :, 0] == 1)
        X1_2, Y1_2 =np.where(pred_sg[:,:,0]==2)
        X1_3, Y1_3 = np.where(pred_sg[:, :, 0] == 5)
        X1_4, Y1_4 = np.where(pred_sg[:, :, 0] == 6)
        X1_5, Y1_5 = np.where(pred_sg[:, :, 0] == 0)
        X0=X1_5
        Y0=Y1_5
        X1=[ X1_1, X1_2, X1_3, X1_4]
        Y1=[ Y1_1,Y1_2 ,Y1_3 ,Y1_4]

        # upper skin
        X2, Y2 = np.where(pred_sg[:, :, 0] == 3)

        # upper garment 9 11 14 -> 4
        X3, Y3 = np.where(pred_sg[:, :, 0] == 4)

        new_img=np.zeros(shape=hum_img.shape)


        bg_color=np.average(hum_img[X0,Y0,:])

        for n in range(len(X0)):
            this_x=X0[n]
            this_y = Y0[n]
            new_img[this_x,this_y,:]=bg_color

        for x in range(len(X1)):
            this_x=X1[x]
            this_y = Y1[x]
            new_img[this_x,this_y,:]=hum_img[this_x,this_y,:]

        new_img[X2, Y2, :] = arm[X2, Y2, :]
        new_img[X3, Y3, :]=warp_gar[X3, Y3, :]

        cv.imwrite(output_path ,new_img)

if __name__=="__main__":
    composition()