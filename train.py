# coding:utf-8
import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn import DataParallel
from datetime import datetime
from config import *
from models import model
from dataset import dataset
from utils import *
import collections
import shutil


def main():
    start_epoch = 1
    saver_dir = mk_save(save_dir, cfg_dir)
    logging = init_log(saver_dir)
    _print = logging.info
       
    ################
    # read dataset #
    ################
    trainset, testset, trainloader, testloader = dataloader(data_dir, 0)
    
    ################                                         
    # define model #
    ################ 
    net = model.model()

    ##########                                      
    # resume #
    ##########

    if resume:
        ckpt = torch.load(resume)
        net_dict = net.state_dict()
        pre_dict = {k: v for k, v in ckpt['state_dict'].items() if k in net_dict}   
        net_dict.update(pre_dict)
        net.load_state_dict(net_dict)
        start_epoch = ckpt['epoch'] + 1
        print('resume', start_epoch)
        start_epoch = START_EPOCH

    criterion = nn.CrossEntropyLoss().cuda()
    kd = nn.KLDivLoss().cuda()

    #####################
    # TODO:num of parameters #
    #####################
    params_count(net)

    #############################
    #TODO: OPTIMIZER FOR BN SLIMMING #
    #############################
    slim_params = params_extract(net)
    
    net = DataParallel(net.cuda())

    for epoch in range(start_epoch, 500): 
        """  
        ##########################  train the model  ###############################
        """
        _print('--' * 50)
        net.train()
        for i, data in enumerate(trainloader):
            # warm up Learning rate
            lr = warm_lr(i, epoch, trainloader)

            ########################
            # Define Optimizer #
            ########################
            optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()),
                                        lr=lr, momentum=0.9, weight_decay=WD, nesterov=True) 

            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            optimizer.zero_grad()
            L1_norm = 0.

            logits = net(img)  

            #############
            # L1 penaty #
            #############
            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
            loss = criterion(logits, label) + LAMBDA1 * L1_norm        
            loss.backward()
            optimizer.step()

            progress_bar(i, len(trainloader), 
                            loss.item(),
                            L1_norm, 
                            lr,msg='train')

        ##########################  evaluate net and save model  ###############################
        if epoch % SAVE_FREQ == 0:
            """
            # evaluate net on train set  
            """
            net.eval()
            train_loss, train_correct, total = test(trainloader)

            train_acc = float(train_correct) / total
            train_loss = train_loss / total
            total_loss = total_loss / total

            _print(
                'epoch:{} - train_loss: {:.4f}, train acc: {:.4f}, L1:{:.4f}, lr:{:.6f}, total sample: {}'.format(
                    epoch,
                    train_loss,
                    train_acc,
                    L1_norm,              
                    lr,                
                    total))

            """
            # evaluate net on test set  
            """
            test_loss, test_correct, total = test(testloader)

            test_acc = float(test_correct) / total
            test_loss = test_loss / total
            _print(
                'epoch:{} - test loss: {:.4f} and test acc: {:.4f} total sample: {}'.format(
                    epoch,
                    test_loss,
                    test_acc,
                    total))

            ##########################  save model  ###############################
            net_state_dict = net.module.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'L1': L1_norm,
                'state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
    print('finishing training')

def warm_lr(i, epoch, dataloader):
        # warm up Learning rate
    lr = 0
    if epoch <= 5:
        lr = LR / (len(dataloader) * 5) * (i + len(dataloader) * epoch)
    elif epoch > 5 and epoch <= 10:
        lr = LR
    elif epoch > 10 and epoch <= 50:
        lr = LR /10
    elif epoch > 50 and epoch <= 90:
        lr = LR * 1e-2
    elif epoch > 90 and epoch <= 130:
        lr = LR * 1e-3
    elif epoch > 130 and epoch <= 170:
        lr = LR * 1e-3 - (LR*1e-3 - LR*1e-4) / 40.  * (epoch - 130.)
    else:
        lr =lr   
    return lr

def test(dataloader):
    loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            logits = net(img)
            # calculate loss
            loss = criterion(logits, label)
            # calculate accuracy
            _, predict = torch.max(logits, 1)
            total += batch_size
            correct += torch.sum(predict.data == label.data)
            loss += loss.item() * batch_size
            
            progress_bar(i, len(dataloader), loss / (i+1), msg='eval train set')

    return loss, correct, total

if __name__ == '__main__':
  main()