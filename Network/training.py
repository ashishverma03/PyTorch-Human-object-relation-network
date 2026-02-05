import pdb
from matplotlib.pyplot import axis
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
import os
from myutils import get_ap_score
import numpy as np
import logging
import time
import pickle
from sklearn.metrics import average_precision_score, accuracy_score 


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
      m.eval()

def train_model(model, device, criterion, optimizer, lr_annealing, train_data, eval_metric, epoch, epoch_size): #, scheduler
    tr_loss_list, tr_map_list = [], []


    print("-------Epoch {}----------".format(epoch+1))
    
    # optimizer.step()
    # scheduler.step()
    
    running_loss = 0.0
    total_loss = 0.0
    running_ap = 0.0
    log_interval = 100
    
    m = torch.nn.Sigmoid()
    soft = torch.nn.Softmax(dim=1)
    
    model.train()  # Set model to training mode
    model.apply(set_bn_eval)

    lr_warmup = float(-1)
    base_lr = float(3e-5)

    # print(optimizer.state_dict())

    for i, batch in enumerate(train_data):


        if i % log_interval == 0:                           #NEW
            new_lr = lr_annealing(epoch * epoch_size + i)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            print(new_lr)
            
        data = batch[0]
        label = batch[1]
        box = batch[2]
        gt_label = label[:, :, 4:5].squeeze(axis=-1)
        gt_label = gt_label.long()
        gt_label = gt_label.squeeze(0)
        # gt_label = gt_label.reshape(1)
        gt_box = label[:, :, :4]
        data, gt_label, gt_box, box = data.to(device), gt_label.to(device), gt_box.to(device), box.to(device)
        optimizer.zero_grad()
        cls_pred = model(data, gt_box,box)
        cls_pred = cls_pred.squeeze(0)
        loss = criterion(cls_pred, gt_label)
        #print("LOSS: ",loss)
        running_loss += loss.item() # sum up batch loss
        total_loss += loss
        # print("RUNNING LOSS: ",running_loss)
        running_ap += get_ap_score(gt_label.to('cpu').detach().numpy(), soft(cls_pred).to('cpu').detach().numpy()) 
        loss.backward()

        # torch.nn.utils.clip_grad_value_(model.parameters(), 5) #NEW
        
        optimizer.step()

        # if epoch!= 0 and epoch!=1:
        #     scheduler.step()
        #     print(scheduler.state_dict())

        if i % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

        # if i%10==0:
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
        #     running_loss = 0.0
            

    train_loss = total_loss/len(train_data)
    running_ap_train = running_ap/len(train_data)
    print("Train Loss: ", train_loss)
    print("Train MAP: ", running_ap_train)
    tr_loss_list.append(train_loss.to('cpu').detach().numpy().item())
    tr_map_list.append(running_ap_train)

    return ([tr_loss_list, tr_map_list])


def val_model(model, device, criterion, optimizer, val_data, save_dir, model_num, log_file, eval_metric, epoch, best_val_map): #, scheduler
    #-------------------------Validation---------------------------#

    val_loss_list, val_map_list = [], []
    
    model.eval()

    eval_metric.reset()
    log_file.write("Epoch {} >>".format(epoch+1))
    total_val_loss = 0
    running_ap_val = 0
    # soft = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in val_data:
            cls_scores = []
            gt_classes = []
            data = batch[0]

            label = batch[1]
            box = batch[2]
            data, label, box = data.to(device), label.to(device), box.to(device)
            gt_box = label[:, :, :4]
            gt_label = label[:,:,4:5].squeeze(axis = -1)
            gt_label = gt_label.long()
            gt_label = gt_label.squeeze(0)
            cls_score = model(data, gt_box, box)
            # pdb.set_trace()
            cls_score_for_loss = cls_score.squeeze(0)
            val_loss = criterion(cls_score_for_loss, gt_label)
            total_val_loss += val_loss
            running_ap_val += get_ap_score(gt_label.to('cpu').detach().numpy(), F.softmax(cls_score_for_loss, dim=-1).to('cpu').detach().numpy())

            
            cls_score = F.softmax(cls_score, dim=-1)
            cls_scores.append(cls_score[:, :, :])
            gt_classes.append(label[:, :, 5:])

            # pdb.set_trace()

            # update metric
            for score, gt_class in zip(cls_scores, gt_classes):
                eval_metric.update(score, gt_class)

        val_loss = total_val_loss/len(val_data)
        validation_map = running_ap_val/len(val_data)
        print("Total Validation Loss: ", val_loss)
        print("Validation mean average precision: ", validation_map) 
        val_msg = '\n'.join(["Validation Loss: {} ".format(val_loss), "Validation MAP: {} ".format(validation_map)])
        log_file.write('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))


        val_loss_list.append(val_loss.to('cpu').detach().numpy().item())
        val_map_list.append(validation_map)

        names, values = eval_metric.get()
        val_loss = total_val_loss/len(val_data)

        if values[-1] >= best_val_map:
            # pdb.set_trace()
            best_val_map = values[-1]
            log_file.write("Saving best weights...\n")
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, os.path.join(save_dir,"model-{}.pth".format(model_num)))
        else:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']# - g['lr']*0.5

        names, values = eval_metric.get()
        val_loss = total_val_loss/len(val_data)
                
    return names, values, val_loss, best_val_map


def test_model(model, device, criterion, val_data, eval_metric): 
    #-------------------------Validation---------------------------#

    
    model.eval()

    eval_metric.reset()
    total_val_loss = 0
    running_ap_val = 0
    cls_scores = []
    image_list = []
    # soft = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in val_data:
            cls_score = []
            # gt_classes = []
            data = batch[0]

            label = batch[1]
            
            box = batch[2]
            img_name = batch[3]
            data, label, box = data.to(device), label.to(device), box.to(device)
            gt_box = label[:, :, :4]
        
            cls_score = model(data, gt_box, box)

            cls_score_for_loss = cls_score.squeeze(0)
            # pdb.set_trace()
            cls_score = F.softmax(cls_score, dim=-1)
            cls_score = cls_score.cpu().numpy()
            cls_scores.append(cls_score[:, :, :])

            image_list.append(img_name)

        with open("test_scores", "wb") as fp:
            pickle.dump(cls_scores, fp)

        with open("image_test", "wb") as fl:
            pickle.dump(image_list, fl)
            




def validate(model, val_data, eval_metric, device):
    """Test on validation dataset."""
    #pdb.set_trace()
    model.train(True)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    val_loss = 0
    eval_metric.reset()
    soft = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch in val_data:
            cls_scores = []
            gt_classes = []
            for data, label, box in zip(*batch):
                data, label, box = data.to(device), label.to(device), box.to(device)
                label = label.unsqueeze(0)
                gt_box = label[:, :, :4]
                box = box.unsqueeze(0)
                data = data.unsqueeze(0)
                # get prediction results
                #pdb.set_trace()
                cls_score = model(data, gt_box, box)
                cls_score_for_loss = cls_score.squeeze(0)
                val_loss += criterion(cls_score_for_loss, label[:, :, 4:5].squeeze(axis = -1).long().squeeze(0))
                # shape (B, N, C)
                cls_score = soft(cls_score)
                cls_scores.append(cls_score[:, :, :])
                gt_classes.append(label[:, :, 5:])

            # update metric
            for score, gt_class in zip(cls_scores, gt_classes):
                eval_metric.update(score, gt_class)
    #pdb.set_trace()
    names, values = eval_metric.get()
    val_loss = val_loss/len(val_data)
    return names, values, val_loss