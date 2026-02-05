from os import name
import argparse
import pdb
from tabnanny import check
from matplotlib.pyplot import axis
from numpy import mod
#from parso import parse
import numpy as np
# from cv2 import transform
import torch
import logging
import os
import time
from myutils import encode_labels,plot_history
from Network.training import train_model , validate, val_model

from torch.utils.data import DataLoader
from Loader import metric,voc_fetcher


from Loader.transformations import HORelationDefaultTrainTransform,HORelationDefaultValTransform
import torchvision

from Network import ho_relation



lr = [3e-1, 1e-2]

# self._root = os.path.join(os.path.expanduser(root), 'VOC2012')



num = 0

if __name__ == '__main__':
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    use_cuda = 1
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")


    model_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),"models")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    # Dataloader
    train_dataset = voc_fetcher.VOCAction(split='train', augment_box=True, load_box=True,transform=HORelationDefaultTrainTransform(600,1000))
    val_dataset = voc_fetcher.VOCAction(split='val', load_box=True,transform=HORelationDefaultValTransform(600,1000))
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True,drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=False,drop_last=False)
    eval_metric = metric.VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, voc_action_type=True)

    lr_annealing = metric.CosineAnnealingSchedule(min_lr=1e-6, max_lr=3e-5, cycle_length=30000)

    epoch_size = train_data.__len__()
    



    #Model
    net = ho_relation.Custom_Model(pretrained=True,device=device).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-5, betas=(0.9, 0.98))
    # optimizer = torch.optim.SGD(list(net.parameters()), lr= 3e-5, momentum= 0.9 , dampening=0.98, weight_decay=5e-4, nesterov=False)
    # optimizer = torch.optim.SGD([   
    #         {'params': list(net.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
    #         {'params': list(net.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    #         ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_data.__len__()*20, eta_min=1e-6, last_epoch=-1)
    
    #Load the weights
    weights_file_path =  os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        # pdb.set_trace()
        checkpoint = torch.load(weights_file_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    #Training
    train_loss_list = []
    train_ap_list = []
    val_loss_list = []
    val_ap_list = []
    log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
    log_file.write("----------Experiment {}-----------\n".format(num))
    log_file.write("Learning Rate: {}".format(lr))
    # pdb.set_trace()
    best_val_map = 0.0
    for epoch in range(0,20):
    
        trn_hist = train_model(net, device, criterion, optimizer , lr_annealing, train_data, eval_metric, epoch, epoch_size) 
       
        train_loss_list.append(trn_hist[0][0])
        train_ap_list.append(trn_hist[1][0])
        
        # print(scheduler.get_lr())
        map_name, mean_ap, val_loss, best_map = val_model(net, device, criterion, optimizer, val_data, model_dir, num, log_file, eval_metric, epoch, best_val_map) 
        # map_name, mean_ap, val_loss = validate(net, val_data, eval_metric, device)
        best_val_map = best_map
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        log_file.write('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        print(val_msg)
        current_map = float(mean_ap[-1])
        val_loss_list.append(val_loss.to('cpu').detach().numpy().item())
        val_ap_list.append(current_map)
        outputs_file_name = '{:s}_{:04d}_val_outputs.csv'.format('', epoch)
        eval_metric.save(file_name=outputs_file_name)
        # torch.save({
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     }, os.path.join(model_dir,"model-{}.pth".format(num)))
        print("Training History: ",trn_hist)
        print("\n")
        # print("Validation History: ",val_hist)
    torch.cuda.empty_cache()
    #pdb.set_trace()
    plot_history(train_loss_list, val_loss_list, "Loss", os.path.join(model_dir, "loss-{}".format(num)))
    plot_history(train_ap_list, val_ap_list, "mAP", os.path.join(model_dir, "accuracy-{}".format(num)))
    log_file.close()