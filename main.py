import torch
import time
from options import Options
from dataset import *
from TPIM_model import *
from PTM_model import *
from RSIM_model import *
import torch.utils.data

def main(opt):

    # train phase
    if opt.mode=="train":

        assert opt.shuffle==True

        # model
        if opt.submodel=="TPIM":
            pgvton_train_dataset = pgvton_dataset()
            pgvton_train_dataset.initialize(opt)
            train_model = sginfer_model(opt)
            train_model.print_network()

        elif opt.submodel=="PIM":
            pgvton_train_dataset = pgvton_dataset()
            pgvton_train_dataset.initialize(opt)
            train_model = viton_model(opt)
            train_model.print_network()

        elif opt.submodel=="RSIM":
            pgvton_train_dataset = armpaint_dataset()
            pgvton_train_dataset.initialize(opt)
            train_model = armpaint_model(opt)
            train_model.print_network()

        # dataset
        pgvton_train_dataset_loader = torch.utils.data.DataLoader(
            pgvton_train_dataset,
            batch_size=opt.batch,
            shuffle=opt.shuffle,
            num_workers=int(opt.nums_works),
            pin_memory=True)

        # model
        time_bench = time.time()
        for i in range(opt.epoch):

            iter = 0
            for index,data in enumerate(pgvton_train_dataset_loader):

                iter+=1

                train_model.setinput(data)
                train_model.forward()

                # vis output
                if iter%opt.vis_ite==0:
                    train_model.vis_result(i,iter)

                # print info
                if iter%opt.log_print_ite==0:
                    train_model.log_print(i,iter,time_bench)

                # vis loss
                if iter%opt.log_vis_ite==0:
                    train_model.plot_loss(i,iter)

            # save model
            if i % opt.save_epo == 0:
                train_model.save_network(i)

            if i==(opt.epoch-1):
                train_model.save_network(i)


    # test phase
    if opt.mode=="eval":

        opt.shuffle= False
        assert opt.inference!=None
        opt.batch = 1
        opt.nums_works =0

        if opt.submodel == "TPIM":
            opt.eval = 'pred_sg'
            pgvton_test_dataset = pgvton_dataset()
            pgvton_test_dataset.initialize(opt)
            test_model = sginfer_model(opt)
            test_model.print_network()

        elif opt.submodel == "PIM":
            opt.eval = 'warped_garment'
            opt.human_mask_dir ='pred_sg\\sg'
            pgvton_test_dataset = pgvton_dataset()
            pgvton_test_dataset.initialize(opt)
            test_model = viton_model(opt)
            test_model.print_network()

        elif opt.submodel == "RSIM":
            opt.eval = 'arm'
            pgvton_test_dataset = armpaint_dataset()
            pgvton_test_dataset.initialize(opt)
            test_model = armpaint_model(opt)
            test_model.print_network()

        # dataset
        test_dataset_loader = torch.utils.data.DataLoader(
            pgvton_test_dataset,
            batch_size=opt.batch,
            shuffle=opt.shuffle,
            num_workers=int(opt.nums_works),
            pin_memory=True)

        # model
        iter = 0
        for index, data in enumerate(test_dataset_loader):
            print(index)
            test_model.setinput(data)
            test_model.eval(iter)

            iter += 1


if __name__=="__main__" :
    opt=Options().parse()
    main(opt)