import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
# noinspection PyUnresolvedReferences
import argparse

from train_code.train_ctpn import  config
from train_code.train_ctpn.ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from train_code.train_ctpn.data.dataset import ICDARDataset

# dataset_download:https://rrc.cvc.uab.es/?ch=8&com=downloads
random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 30
lr = 1e-3
resume_epoch = 0


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'ctpn_ICDAR_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    try:
        torch.save(state, check_path)
    except BaseException as e:
            print(e)
            print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = config.pretrained_weights
    print('exist pretrained ',os.path.exists(checkpoints_weight),checkpoints_weight)
    if os.path.exists(checkpoints_weight):
        pretrained = False

    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    model = CTPN_Model()
    model.to(device)
    
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

    params_to_uodate = model.parameters()
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)
    
    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr: RPN_REGR_Loss = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

########### 用于保存训练时的数据 ############
    Loss_cls = []
    Loss_regr = []
    Loss_total = []
    epoch_line = []
 ########################################
    for epoch in range(resume_epoch+1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#'*50)
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
         #  scheduler.step(epoch)
    
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            # print(imgs.shape)
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()
    
            out_cls, out_regr = model(imgs)
            loss_cls = critetion_cls(out_cls, clss)         # 分类损失（预测每个anchor是否包含文本区域的classification loss)
            loss_regr = critetion_regr(out_regr, regrs)     # 回归损失(文本区域中每个anchor的中心坐标与高度的regression loss; 文本区域两侧anchor的中心坐标的regression loss）
    
            loss = loss_cls + loss_regr  # total loss
            loss.backward()
            optimizer.step()               ################################
    
            epoch_loss_cls += loss_cls.item()   ############取 Loss_cls 的平均值
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i+1


            print(f'Ep:{epoch}/{epochs-1}--'
                  f'Batch:{batch_i}/{epoch_size}\n'
                  f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                  f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
                  f'loss:{epoch_loss/mmp:.4f}\n')

        scheduler.step(epoch)
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')

        #########################
        Loss_cls.append(round(epoch_loss_cls, 4))
        Loss_regr.append(round(epoch_loss_regr, 4))
        Loss_total.append(round(epoch_loss, 4))
        epoch_line.append(epoch)
        print("Loss_cls:", Loss_cls)
        print("Loss_regr:", Loss_regr)
        print("Loss_total:", Loss_total)
        print("epoch_line:", epoch_line)
        file = open('D:/OCR/train_code/train_ctpn/train_loss_recard.txt', 'a+')
        file.write('#############   epoch = ' + str(epoch) + '  ############' + '\n')
        file.write('Loss_regr: ' + str(Loss_regr) + '\n' + 'Loss_cls: ' + str(Loss_cls) + '\n' + 'Loss_total: ' + str(
            Loss_total) + '\n')
        file.close()
        ###############################

        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
