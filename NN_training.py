# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:14:19 2022

@author: scabini
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--datapath', type=str, default= os.getenv("HOME") + '/Experiments/datasets/images/', help='Path to load the dataset')
    parser.add_argument('--output_path', type=str, default= os.getenv("HOME") + '/Experiments/CNN_TL/results/', help='Path for saving models and metrics')    
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use for training (see nvidia-smi)')
    parser.add_argument('--multigpu',  action='store_true', default=False, help='Either to use parallel GPU processing or not')
  
    parser.add_argument('--tl',  action='store_true', default=False, help='Transfer learning or not')
    parser.add_argument('--backbone', type=str, default='convnext_nano', help='Pretrained model name according to timm')
    parser.add_argument('--seed', type=int, default=666, help='Base random seed for weight initialization and data splits/shuffle')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')   
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--K', type=int, default=10, help='K-fold splits')
    parser.add_argument('--target', type=str, default='SVI', help='Target to train on')

    return parser.parse_args()

########################## MAKING EVERYTHING DETERMINISTIC #######
num_workers = 12
import multiprocessing
import os
import numpy as np
import pickle
import copy
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
os.environ["OMP_NUM_THREADS"]=str(num_workers)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
##################################################################
import dataset


if __name__ == "__main__":
    args = parse_args()
  
    gpus = torch.cuda.device_count()
    total_cores=multiprocessing.cpu_count()
    # num_workers = int(total_cores/2)-2
    
    # print("System has", gpus, "GPUs and", total_cores, "CPU cores. Using", num_workers, "cores for data loaders.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 32 
    # imgdimm = (256,192)
    # ImageNet normalization parameters and other img transformations
    averages =  (0.485, 0.456, 0.406)
    variances = (0.229, 0.224, 0.225)  
    # REGULARIZATION/DATA AUG.
    imgdimm = (384, 512)
        
    train_transform = transforms.Compose([
        transforms.ToTensor(),        
        transforms.Resize(imgdimm),
        # transforms.RandomResizedCrop(imgdimm, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
        ###### geometric transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(
            torch.nn.ModuleList([transforms.RandomRotation(180),
                ]), p=0.5),        
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ######
        # Additional transformations
        transforms.RandomApply(
            torch.nn.ModuleList([transforms.ColorJitter(
                brightness=0.2, contrast=0, saturation=0, hue=0),
                ]), p=0.5),        
        ######
        transforms.Normalize(averages, variances),   
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(imgdimm),
        transforms.Normalize(averages, variances),        
    ])
   
    train_dataset = dataset.Microbiological_Images(root=args.datapath, transform=train_transform, label=args.target) 
    test_dataset = dataset.Microbiological_Images(root=args.datapath, transform=val_transform, label=args.target)   
    
    # if args.target == 'bulking':
    #     train_dataset.targets=torch.nn.functional.one_hot(torch.FloatTensor(train_dataset.targets).to(torch.int64))
    #     test_dataset.targets=torch.nn.functional.one_hot(torch.FloatTensor(test_dataset.targets).to(torch.int64))
        
    idx = np.arange(len(train_dataset))
    
    kfold = sklearn.model_selection.KFold(n_splits=args.K, shuffle=True, random_state = args.seed)
    kfold.get_n_splits(idx)
    for fold, (train_index, test_index) in enumerate(kfold.split(idx)):
        torch.manual_seed(args.seed*(fold+1))
        np.random.seed(args.seed*(fold+1))
        
        #for each network, I save its performance metrics in a pkl file
        subfolders = args.output_path + '/' 
        os.makedirs(subfolders, exist_ok=True)
        
        file = subfolders +str(args.seed) + '_' + str(args.K) + '_' + str(fold) + '_' + args.backbone + '_' + str(args.lr) 

        if args.tl:
            file+= '_' + str(args.epochs) + '_batch' + str(BATCH_SIZE) + '_' + args.target + '_TRANSFERLEARNING.pk'
        else:
            file+= '_' + str(args.epochs) + '_batch' + str(BATCH_SIZE) + '_' + args.target + 'FROMSCRATCH.pk'
        
        exists = os.path.isfile(file) 
        # print(cluster_out_path+file)
        
        converged_on = 0
        convergences = []
        loss_convergences = []
        total_trainacc = []
        total_validacc = []
        total_validloss = []
        total_testacc  = []        
        all_trains_accs = []
        all_trains_loss = []    
        all_valid_accs = []
        all_valid_losses = []    
        all_test_accs = []
        all_test_losses = [] 
        all_gradients = []    
        train_accs, train_losses, val_accs, val_losses = [], [], [], []
        gradients = []
        net_time = time.time()
        if exists:
            with open(file, 'rb') as f:
               train_accs, train_losses, val_accs, val_losses, converged_on, predictions, dates = pickle.load(f)    
            
        else: 
                  
            train = torch.utils.data.Subset(train_dataset, train_index)
            test = torch.utils.data.Subset(test_dataset, test_index)
         
            if args.target == 'bulking':
                TARGET_SCALE=1
            else:
                TARGET_SCALE = max([train_dataset.targets[indd] 
                                    for indd in train_index])
            # target_weights = inbalance_weights([train_dataset.targets[indd] 
            #                                     for indd in train_index])
            
            trainloader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True,
                                                      drop_last=False, pin_memory=True, num_workers=num_workers)
            
            testloader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False,
                                                     drop_last=False, num_workers=num_workers)

            print(args.backbone, ' - train/test ratio:', len(train), len(test), 'transfer learning?', args.tl)
            print('learning rate:', args.lr, ', target scale:', TARGET_SCALE)
            
            if args.target == 'bulking':
                criterion = nn.CrossEntropyLoss() #
                n_classes = 2 #here we do binary classification: bulking or not                
                 
            else:            
                criterion = nn.MSELoss() #mse for regression
                n_classes = 1 #do regression in this case
                
            if args.tl:                
                if 'convnext_nano' in args.backbone or 'convnext_tiny' in args.backbone or 'resnet18' in args.backbone:
                    drop_path_rate=0.1
                else:
                    drop_path_rate=0.2
                    
                layer_decay = 0.8
                weight_decay=1e-8
                if 'resnet' in args.backbone:
                    model = timm.create_model(args.backbone, pretrained=True, 
                                              num_classes=n_classes,
                                              drop_path_rate=drop_path_rate)
                elif 'convnext' in args.backbone:
                    model = timm.create_model(args.backbone, pretrained=True, 
                                              num_classes=n_classes, head_init_scale=0.001,
                                              drop_path_rate=drop_path_rate)
                else:
                    model = timm.create_model(args.backbone, pretrained=True, 
                                              num_classes=n_classes)
            else:
                if 'convnext_nano' in args.backbone or 'convnext_tiny' in args.backbone or 'resnet18' in args.backbone:
                    drop_path_rate=0.2
                else:
                    drop_path_rate=0.4
                    
                layer_decay = 1.0
                weight_decay= 1e-3
                if 'resnet' in args.backbone:
                    model = timm.create_model(args.backbone, pretrained=False, 
                                              num_classes=n_classes,
                                              drop_path_rate=drop_path_rate,
                                              )
                elif 'convnext' in args.backbone:
                    model = timm.create_model(args.backbone, pretrained=False, 
                                              num_classes=n_classes, head_init_scale=1.0,
                                              drop_path_rate=drop_path_rate,
                                              )
                else:
                    model = timm.create_model(args.backbone, pretrained=False, 
                                              num_classes=n_classes)
   
            # for param in model.parameters():
            #     param.requires_grad = False 
            # for param in model.fc.parameters():
            #     param.requires_grad = True
                
            num_layers=3
            layer_decay= list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            
            if 'resnet' in args.backbone:
                layer_wise_lr=[ {"params": model.conv1.parameters(), "lr": args.lr*layer_decay[0]},
                                {"params": model.layer1.parameters(), "lr": args.lr*layer_decay[0]},
                                {"params": model.layer2.parameters(), "lr": args.lr*layer_decay[1]},
                                {"params": model.layer3.parameters(), "lr": args.lr*layer_decay[2]},
                                {"params": model.layer4.parameters(), "lr": args.lr*layer_decay[3]},
                                {"params": model.fc.parameters(), "lr": args.lr*layer_decay[4]},
                                ]
            elif 'convnext' in args.backbone:
                layer_wise_lr=[ {"params": model.stages[0].parameters(), "lr": args.lr*layer_decay[0]},
                                {"params": model.stages[1].parameters(), "lr": args.lr*layer_decay[1]},
                                {"params": model.stages[2].parameters(), "lr": args.lr*layer_decay[2]},
                                {"params": model.stages[3].parameters(), "lr": args.lr*layer_decay[3]},
                                {"params": model.head.parameters(), "lr": args.lr*layer_decay[4]},
                                ]
            else:
                print("!!! Layer-wise learning rates not available for the given network")
                layer_wise_lr=model.parameters()
            
            optimizer = torch.optim.AdamW(layer_wise_lr,                                          
                                          lr=args.lr,  weight_decay=weight_decay) #standard lr=0.001
            
            # learning rate update scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=args.epochs,
                                                                   eta_min=0, last_epoch= -1)
            scaler = torch.cuda.amp.GradScaler()
            
            if args.multigpu:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
            
            best_model=copy.deepcopy(model)    
            model.to(device)
            
            for epoch in range(args.epochs):
                torch.manual_seed(args.seed*(fold+1) + epoch)
                np.random.seed(args.seed*(fold+1) + epoch)
            # for epoch in range(args.epochs):  # loop over the dataset multiple times
                start_time = time.time()
                
                # if epoch == args.epochs//5:
                #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr/100,  weight_decay=0.0005) #standard lr=0.001
                #     # learning rate update scheduler
                #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                #                                                            T_max=(len(trainloader)*(args.epochs-(args.epochs//5)))//2,
                #                                                            eta_min=0, last_epoch= -1)
                #     for param in model.parameters():
                #         param.requires_grad = True
                
                running_loss = []
                batch_grads = []
                model.train()
                for i, data in enumerate(trainloader, 0):
                    # weights = torch.FloatTensor([target_weights[t] for t in
                    #                              data[1].cpu().detach().numpy()]).to(device)
                    # weights = target_weights[data[1].cpu().detach().numpy()].to(torch.float32).to(device)
                    
                    # get the inputs; data is a list of [inputs, labels]
                    # if args.large_scale:#in the large scale case, have to put batch on GPU
                    if args.target == 'bulking':
                        inputs, labels = data[0].to(device), data[1].to(torch.int64).to(device)
                    else:
                        inputs, labels = data[0].to(device), data[1].to(torch.float32).to(device)
                        labels = labels.unsqueeze(1)/TARGET_SCALE
                    # else: #in the small scale case, the batches are already preloaded on GPU
                    # inputs, labels = data[0], data[1]
            
                    # zero the parameter gradients
                    optimizer.zero_grad()  
                    with torch.cuda.amp.autocast():
                        output = model(inputs) 
                        # print(output.size(), labels.size())
                        loss = criterion(output, labels)
                        # loss = criterion_weighted(output, labels, weight=weights)
                        
                    scaler.scale(loss).backward()                    
                    scaler.unscale_(optimizer)
                    # nn.utils.clip_grad_norm_(model.parameters(), 10.0)                    
                    scaler.step(optimizer)
                    scaler.update()  
                    running_loss.append(loss.item()) 
                    
                    # ave_grads = []                    
                    # for n, p in model.named_parameters():
                    #     if(p.requires_grad) and ("bias" not in n):
                    #         # try:
                    #         ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                    #         # except:
                    #         #     ave_grads.append(0)
                    # batch_grads.append(ave_grads)
                
                scheduler.step()
                # gradients.append(np.mean(batch_grads, axis=0))
                
                tlosss = np.mean(running_loss)
                tlosss = tlosss*TARGET_SCALE 
                train_losses.append(tlosss)
                
                vlo_bff = []
                with torch.no_grad():
                    model.eval()
                    for ii, data in enumerate(testloader, 0):
                        if args.target == 'bulking':
                            inputs, labels = data[0].to(device), data[1].to(torch.int64).to(device)
                        else:
                            inputs, labels = data[0].to(device), data[1].to(torch.float32).to(device)
                            labels = labels.unsqueeze(1)/TARGET_SCALE

                        outputs = model(inputs)
                        loss = criterion(outputs, labels) #no weighting loss during validation
                        vlo_bff.append(loss.item())                        
                    
                vlo = np.mean(vlo_bff)
                vlo = vlo*TARGET_SCALE
                # vlo = vlo * 1.4942304976316982 + 11.537147406898626 
                #define here the convergence criterion. I considered accuracy on the paper
                if len(val_losses) == 0 or vlo < min(val_losses): 
                    best_model = copy.deepcopy(model)                       
                    # best_model = model
                    converged_on= epoch
                        
                val_losses.append(vlo)
                
                train_time = np.round(time.time() - start_time, decimals=4)
                
                if epoch+1 < 10:
                    perfect_string = 'fold ' + str(fold) + ", epoch:  " + str(epoch+1) + '/' + str(args.epochs)
                else:
                    perfect_string = 'fold ' + str(fold) + ", epoch: " + str(epoch+1) + '/' + str(args.epochs)
                    
                print(perfect_string,
                      f"loss: {np.round(train_losses[-1], 4):.4f}, "
                      # f"acc: {np.round(train_accs[-1] * 100, 2):.2f}%", 
                          f"val loss: {np.round(val_losses[-1], 4):.4f}, "
                          f"lr: {scheduler.get_last_lr()[0]:.7f}",
                          # 'gradients=(', np.min(batch_grads), np.mean(batch_grads), np.max(batch_grads), ')'
                          "||", np.round(train_time, decimals=2), 'sec')
                
            del model
            model = []            
            predictions = []
            dates = [test_dataset.dates[i] for i in test_index]
            best_model.eval()
            for ii, data in enumerate(testloader, 0):
                if args.target == 'bulking':
                    inputs, labels = data[0].to(device), data[1].to(torch.int64).to(device)
                    
                    predictions.append((labels.cpu().detach().numpy()[0],
                                        best_model(inputs).cpu().detach().numpy()[0,0], best_model(inputs).cpu().detach().numpy()[0,1]))
                    
                else:
                    inputs, labels = data[0].to(device), data[1].to(torch.float32).to(device)
                    labels = labels.unsqueeze(1)/TARGET_SCALE
                    
                    predictions.append((labels.cpu().detach().numpy()[0],
                                        best_model(inputs).cpu().detach().numpy()[0,0]*TARGET_SCALE))
                
            with open(file, 'wb') as f:
                pickle.dump([train_accs, train_losses, val_accs, val_losses, 
                              converged_on, predictions, dates], f)
                
            torch.save(best_model.state_dict(), file + '_NETWORK.pt')

            
            del best_model
            best_model= []
            
        # print(predictions)
            
        
        convergences.append(converged_on+1)
        loss_convergences.append(np.argmin(val_losses)+1)
        total_validloss.append(val_losses[np.argmin(val_losses)])
        
        # all_gradients.append(gradients)
        all_trains_loss.append(train_losses)
        all_valid_losses.append(val_losses)
        
        net_time = np.round(time.time() - net_time, decimals=4)
        
        print('fold',fold, ", ep:", converged_on+1, "|" ,
                f"train loss: {round(train_losses[converged_on], 4):.4f}, "
                  # f"train acc: {round(train_accs[converged_on] * 100, 2):.2f}%", 
                       f"val loss: {round(val_losses[converged_on], 3):.3f}, "
                      # f"val acc: {round(val_accs[converged_on] * 100, 2):.2f}%",   
                      # f"test loss: {round(tl, 3):.3f}, "
                      # f"test acc: {round(tac * 100, 2):.2f}%"
                      " || ", net_time, 'sec')        

            