import pandas as pd
import argparse
import numpy as np
import random 
import torch 
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import image
import logging
import wandb

from dataloader import * 
from models import * 
from predict import * 


#helper functions: 
def compute_group_avg(n_groups, losses, group_idx, device):
    # compute observed counts and mean loss for each group
    #rint('losses_size', losses.size(0))
    group_map = (group_idx == torch.arange(n_groups).unsqueeze(1).long()).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count==0).float() # avoid nans
    #print('group count sum', group_count.sum())
    group_loss = (group_map.to(device) @ losses.view(-1))/group_denom.to(device) #check loss dim 
    return group_loss, group_count

def compute_robust_loss(n_groups, step_size, group_loss, device): 
    #they either normalise group loss or adjust group loss by adding adj/sqrt(count)
    group_probs = torch.ones(n_groups).cuda() / n_groups
    group_probs = group_probs * torch.exp(step_size * group_loss)
    group_probs = group_probs/group_probs.sum()
    robust_loss = group_loss.to(device) @ group_probs.to(device)
    return robust_loss

def start_training_robust(args,model, train_dataloader, val_dataloader, device):
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    aggregate_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
   # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    sigmoid = torch.nn.Sigmoid()
    
    r_training_losses=[]
    training_losses=[]
    validation_losses=[]
    validation_accuracy=[]
    training_accuracy=[]
    step_size=args.STEP_SIZE
    #validation_metrics={}

    train_group_losses_df = pd.DataFrame({'epoch':[]})
    val_group_losses_df = pd.DataFrame({'epoch':[]})
    

    auroc_metric = BinaryAUROC(thresholds=None)
    f1_metric = BinaryF1Score()

    min_val_loss = float('inf')
    max_val_acc = -float('inf')
    max_sns_acc = -float('inf')

    true_groups=[0,1,2,3] #there are 4 true groups 
    
    print("training started")
    for epoch in range(args.MAX_EPOCHS): 
        ########################## TRAINING ###################################
        model.train()
        r_train_loss=0
        train_loss=0
        correct=0
        total=0

        train_losses_by_group = {i: [] for i in range(args.n_groups)}
        train_loss_by_group = {i: 0 for i in range(args.n_groups)}
        train_losses_by_truegroup = {i: [] for i in range(4)}

        train_preds_by_group = {i: [] for i in range(args.n_groups)}
        train_target_by_group = {i: [] for i in range(args.n_groups)}
        train_auroc_by_group = {i: [] for i in range(args.n_groups)}
        train_f1_by_group = {i: [] for i in range(args.n_groups)}

        
        train_predictions = pd.DataFrame({'group':[],'predictions':[],'preds':[],'gt':[]})
        val_predictions = pd.DataFrame({'group':[],'predictions':[],'preds':[],'gt':[]})
        
        train_group=[]
        train_predicted=[]
        train_gt=[]
        train_preds=[]
        train_path=[]
        train_truelabel=[]

        all_preds=[]
        all_labels=[]
        
        for path, batch_x, batch_y,l,truegroup in train_dataloader: #16, 100, 512, 512
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x.float())
            pred = pred.squeeze(1)
            
            predicted = (sigmoid(pred)>0.5).float()
            correct += (predicted == batch_y).sum().item() 
            total+=batch_y.size(0)

            loss = criterion(pred, batch_y.float()) 

            #g = torch.tensor(list(l)) #l is originally a tuple 
    
            group_loss, group_count = compute_group_avg(args.n_groups,loss, l, device)

            # print(f"Epoch {epoch+1} : Group loss = {group_loss.detach().cpu().numpy()}, Group Count = {group_count.detach().cpu().numpy()}, Group Sum = {group_count.sum().cpu().numpy()}")

            robust_loss = compute_robust_loss(args.n_groups, step_size, group_loss, device)

            r_train_loss+=robust_loss.item()
            train_loss+=loss.mean().item()


            unique_groups=torch.unique(l)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

            for g in unique_groups:
                mask = l == g
                pred_split = pred[mask]
                target_split = batch_y[mask]
                train_losses_by_group[g.item()].append(group_loss[g.item()].item())
                train_preds_by_group[g.item()].extend(pred_split.cpu().tolist())
                train_target_by_group[g.item()].extend(target_split.cpu().tolist())
                #print(train_losses_by_group[g.item()])
            
            for g in true_groups:
                mask = truegroup == torch.tensor(g)
                pred_split = pred[mask]
                target_split = batch_y[mask]
                predicted_split = (sigmoid(pred_split)>0.5).float()
                loss = aggregate_criterion(pred_split, target_split.float())
                train_losses_by_truegroup[g].append(loss.item())
                                
            train_group.extend(l.tolist())
            train_predicted.extend(predicted.tolist())
            train_gt.extend(batch_y.tolist())
            train_preds.extend(pred.tolist())
            train_path.extend(path)
            train_truelabel.extend(truegroup.tolist())
            
            optimizer.zero_grad()
            
            robust_loss.backward()
            optimizer.step()
            #print('iteration done')
            
        #print(f'epoch {epoch}, train done')
        train_acc = correct / total
        training_accuracy.append(train_acc)
        r_train_loss = r_train_loss/len(train_dataloader)
        train_loss = train_loss/len(train_dataloader)
        all_preds= torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        train_auroc = auroc_metric(all_preds, all_labels)

        unique_groups = np.unique(np.array(train_group))

        for g in unique_groups:
            train_preds_by_group[g.item()]= torch.tensor(train_preds_by_group[g.item()])
            train_target_by_group[g.item()] = torch.tensor(train_target_by_group[g.item()]) 
            train_loss_by_group[g.item()] = np.mean(train_losses_by_group[g.item()])
            print(f'train loss = {epoch}, group {g.item()} - ', train_loss_by_group[g.item()])
            train_group_losses_df.loc[epoch, str(g.item())] = train_loss_by_group[g.item()]
            train_group_losses_df.loc[epoch, 'epoch'] = epoch
            train_auroc_by_group[g.item()]=auroc_metric(train_preds_by_group[g.item()], train_target_by_group[g.item()])
            train_f1_by_group[g.item()]=f1_metric(train_preds_by_group[g.item()], train_target_by_group[g.item()])
        
        r_training_losses.append(r_train_loss)
        training_losses.append(train_loss)

        train_predictions['group']=train_group
        train_predictions['predictions']=train_predicted
        train_predictions['gt']=train_gt
        train_predictions['preds']=train_preds
        train_predictions['train_path']=train_path
        train_predictions['true_group']=train_truelabel
        
        train_group_accuracies = train_predictions[['group','predictions','gt','preds']].groupby('group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))

        train_true_group_accuracies = train_predictions[['true_group','predictions','gt','preds']].groupby('true_group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))

        ########################## VALIDATION ###################################
        
        val_loss=0
        correct=0
        total=0
        all_preds=[]
        all_labels=[]

        val_losses_by_group = {i: [] for i in range(args.n_groups)}
        val_loss_by_group = {i: 0 for i in range(args.n_groups)}
        val_auroc_by_group = {i: [] for i in range(args.n_groups)}
        val_f1_by_group = {i: [] for i in range(args.n_groups)}
        
        preds_by_group = {i: [] for i in range(args.n_groups)}
        target_by_group = {i: [] for i in range(args.n_groups)}

        val_losses_by_truegroup = {i: [] for i in range(4)}

        val_group=[]
        val_predicted=[]
        val_preds=[]
        val_gt=[]
        val_path=[]
        val_truelabel=[]

        print('val loop')
        
        with torch.no_grad():
            model.eval()
            
            for path, batch_x, batch_y,l,truegroup in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x.float()).squeeze(1)
                
                predicted = (sigmoid(pred)>0.5).float()
                correct += (predicted == batch_y).sum().item() 
                total+=batch_y.size(0)
                loss = criterion(pred, batch_y.float())
                
                #get robust loss: 
                group_loss, group_count = compute_group_avg(args.n_groups, loss, l, device)
                robust_loss = compute_robust_loss(args.n_groups, step_size, group_loss, device)
                
                val_loss += robust_loss.item()
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

                unique_groups = torch.unique(l)
                
                for g in unique_groups:
                    mask = l == g
                    pred_split = pred[mask]
                    target_split = batch_y[mask]
                    predicted_split = (sigmoid(pred_split)>0.5).float()
                    preds_by_group[g.item()].extend(pred_split.cpu().tolist())
                    target_by_group[g.item()].extend(target_split.cpu().tolist())
                    val_losses_by_group[g.item()].append(group_loss[g.item()].item())

                for g in true_groups:
                    mask = truegroup == torch.tensor(g)
                    pred_split = pred[mask]
                    target_split = batch_y[mask]
                    predicted_split = (sigmoid(pred_split)>0.5).float()
                    loss = aggregate_criterion(pred_split, target_split.float())
                    val_losses_by_truegroup[g].append(loss.item())
                    
            
                val_group.extend(l.tolist())
                val_predicted.extend(predicted.tolist())
                val_gt.extend(batch_y.tolist())
                val_preds.extend(pred.tolist())
                val_path.extend(path)
                val_truelabel.extend(truegroup.tolist())
            
            unique_groups = np.unique(np.array(val_group))
            
            for g in unique_groups:
                preds_by_group[g.item()]= torch.tensor(preds_by_group[g.item()])
                target_by_group[g.item()] = torch.tensor(target_by_group[g.item()]) 

                val_f1_by_group[g.item()]=f1_metric(preds_by_group[g.item()], target_by_group[g.item()])
                val_auroc_by_group[g.item()]=auroc_metric(preds_by_group[g.item()], target_by_group[g.item()])
                
                val_loss_by_group[g.item()] = np.mean(val_losses_by_group[g.item()])
                #print(f'val loss = {epoch} group {g.item()} - ', val_loss_by_group[g.item()])
                
                val_group_losses_df.loc[epoch, str(g.item())] = val_loss_by_group[g.item()]
                val_group_losses_df.loc[epoch, 'epoch'] = epoch
                
                
            val_loss = val_loss / len(val_dataloader)
            val_acc = correct / total 
            validation_losses.append(val_loss)
            validation_accuracy.append(val_acc)
            
            all_preds= torch.tensor(all_preds)
            all_labels = torch.tensor(all_labels) 

            val_f1 = f1_metric(all_preds, all_labels)
            val_auroc = auroc_metric(all_preds, all_labels)

            val_predictions['group']=val_group
            val_predictions['predictions']=val_predicted
            val_predictions['gt']=val_gt
            val_predictions['preds']=val_preds
            val_predictions['val_path']=val_path
            val_predictions['true_group']=val_truelabel

            val_group_accuracies = val_predictions[['group','predictions','gt','preds']].groupby('group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))

            val_true_group_accuracies = val_predictions[['true_group','predictions','gt','preds']].groupby('true_group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))
            
            worst_group = val_true_group_accuracies.idxmin() #worst group - worst true group!
            worst_group_accuracy = val_true_group_accuracies[worst_group]
            worst_sns_accuracy = val_group_accuracies[val_group_accuracies.idxmin()]
            print('WORST GROUP ACCURACY = ', worst_group_accuracy)
            print('WORST SNS ACCURACY = ', worst_sns_accuracy)
                
        ########################## LOGGING ###################################
            
        #if (epoch+1)%10 == 0: 
        logging.info(f"Epoch {epoch+1} : Robust loss = {r_train_loss}, Robust Val loss = {val_loss}, Val Acc = {val_acc}, Val AUROC = {val_auroc}, Val F1 = {val_f1}, Worst Group Val Acc = {worst_group_accuracy}")


        wandb.log({"overall_ERM_loss": train_loss, "overall_robust_training_loss":r_train_loss,"overall_robust_validation_loss": val_loss,
            "overall_validation_auroc":val_auroc,"overall_validation_f1":val_f1,"overall_training_accuracy":train_acc, "overall_validation_accuracy":val_acc,
                  "overall_training_auroc":train_auroc,"val_worst_true_group_accuracy":worst_group_accuracy})
        
        for g in unique_groups:
            wandb.log({f"group{g}_validation_auroc":val_auroc_by_group[g.item()],
                       f"group{g}_validation_f1":val_f1_by_group[g.item()],
                       f"group{g}_training_accuracies":train_group_accuracies[g.item()],
                       f"group{g}_validation_acc":val_group_accuracies[g.item()], 
                       f"group{g}_validation_loss":val_loss_by_group[g.item()], 
                       f"group{g}_training_loss":train_loss_by_group[g.item()],
                       f"group{g}_training_auroc":train_auroc_by_group[g.item()]})

        for g in true_groups:
            wandb.log({f"truegroup{g}_validation_accuracy":val_true_group_accuracies[g],
                       f"truegroup{g}_training_accuracy":train_true_group_accuracies[g],
                      f"truegroup{g}_validation_loss":val_losses_by_truegroup[g],
                      f"truegroup{g}_training_loss":train_losses_by_truegroup[g]})
            
            
        group_name = args.GROUP_CNAME
        
        experiment_name = f'seed_{args.RANDOMSEED}+group{group_name}+stepsize_{args.STEP_SIZE}+epoch_{epoch+1}+weightdecay_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}+transform_{args.TRANSFORM}'

        experiment_name_no_epoch = f'seed_{args.RANDOMSEED}+group{group_name}+stepsize_{args.STEP_SIZE}+weightdecay_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}+transform_{args.TRANSFORM}'

        if val_loss<min_val_loss:
            min_val_loss = val_loss
            logging.info(f'{epoch} - minimum val loss! saving weights...')
            torch.save(model, f'{args.OUTDIR}/{experiment_name_no_epoch}_lowest_val_loss.pth')
            train_predictions['epoch']=epoch
            val_predictions['epoch']=epoch
            min_val_loss_epoch = epoch
            # train_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_train_preds_minloss.csv')
            # val_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_val_preds_minloss.csv')

        if worst_group_accuracy>max_val_acc:
            max_val_acc = worst_group_accuracy #true group accuracy!
            print(max_val_acc)
            logging.info(f'{epoch} - maximum val acc! saving weights...')
            torch.save(model, f'{args.OUTDIR}/{experiment_name_no_epoch}_max_val_acc.pth')
            train_predictions['epoch']=epoch
            val_predictions['epoch']=epoch
            max_val_acc_epoch = epoch
            # train_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_train_preds_maxacc.csv')
            # val_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_val_preds_maxacc.csv')

        if worst_sns_accuracy>max_sns_acc:
            max_sns_acc = worst_sns_accuracy #true group accuracy!
            print(max_sns_acc)
            torch.save(model, f'{args.OUTDIR}/{experiment_name_no_epoch}_max_sns_acc.pth')
            train_predictions['epoch']=epoch
            val_predictions['epoch']=epoch
            max_sns_acc_epoch = epoch 
            #train_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_train_preds_maxsns.csv')
            #val_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_val_preds_maxsns.csv')
            
        if (epoch+1)%10 == 0:
            logging.info('saving weights...')
            torch.save(model, f'{args.OUTDIR}/models/{epoch}_{args.n_groups}_model_weights.pth')
            #train_predictions.to_csv(f'{args.OUTDIR}/train_preds/{epoch}_{args.n_groups}_train_preds.csv')
            #val_predictions.to_csv(f'{args.OUTDIR}/val_preds/{epoch}_{args.n_groups}_val_preds.csv')
    
    # np.save(f'{args.OUTDIR}/losses/train_loss_{experiment_name}.npy', np.array(training_losses))
    # np.save(f'{args.OUTDIR}/losses/robust_train_loss_{experiment_name}.npy', np.array(r_training_losses))
    # np.save(f'{args.OUTDIR}/losses/robust_val_loss_{experiment_name}.npy', np.array(validation_losses))
    # np.save(f'{args.OUTDIR}/accuracies/val_accuracy_{experiment_name}.npy', np.array(validation_accuracy))
    # np.save(f'{args.OUTDIR}/accuracies/val_accuracy_{experiment_name}.npy', np.array(training_accuracy))
    
    # val_group_losses_df.to_csv(f'{args.OUTDIR}/val_group_losses.csv')
    # train_group_losses_df.to_csv(f'{args.OUTDIR}/train_group_losses.csv')

    print('done.')

    model_minloss_path = f'{args.OUTDIR}/{experiment_name_no_epoch}_lowest_val_loss.pth'
    model_maxacc_path = f'{args.OUTDIR}/{experiment_name_no_epoch}_max_val_acc.pth'
    model_snsmaxacc_path = f'{args.OUTDIR}/{experiment_name_no_epoch}_max_sns_acc.pth'

    return model_minloss_path, model_maxacc_path,model_snsmaxacc_path
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--PROJECT_NAME", type=str, default = 'stage-2-gdro-training', help="name of project")
    #parser.add_argument("--RUN_NAME", type=str, default = 'run_unnamed', help="name of run")
    parser.add_argument("--RANDOMSEED", type=int, default = 42, help="random seed")

    # Dataloader Arguments
    parser.add_argument("--M_LABEL", type=str, default = 'negbio', help="Labels for mimic: negbio or chexpert")
    parser.add_argument("--IMAGE_SIZE", type=int, default = 256, help="Image resize value")
    parser.add_argument("--TRANSFORM", action="store_true", help = 'Apply transformation? true/false')
    parser.add_argument("--NORMALISE", action="store_true", help='Apply Imagenet normalisation')
    parser.add_argument("--NWORKERS", type=int, default = 8, help="Num Workers")
    parser.add_argument("--BATCH_SIZE", type=int, default = 8, help="Num Workers")
    #parser.add_argument("--TRAIN_SPLIT", type=float, default=0.7, help = "Data train split")
    parser.add_argument("--BORDER_SZ", type=int, default=0, help="Border Size for nuisance")
    parser.add_argument("--K", type=int, default = 10000, help="Sample size from each dataset")
    parser.add_argument("--GROUP_CNAME", type=str, default='sns_group', help="Name of group IDX feature")
    
    # Model Arguments
    parser.add_argument("--n_groups", type=int, default=4, help = "number of groups")
    parser.add_argument("--STEP_SIZE", type=int, default=1, help = "dro step size")
    parser.add_argument("--MAX_EPOCHS", type=int, default=200, help = "Max training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default = 1e-3, help="Learning rate")
    parser.add_argument("--WEIGHT_DECAY", type=float, default = 1e-4, help="Weight Decay")
    parser.add_argument("--OUTDIR", type=str, default="./out-robust", help = "Dir of out files - losses/weights")
    parser.add_argument("--MODEL_TYPE", type=str, default="resnet18", help = "resnet18 or resnet50")
    parser.add_argument("--PRETRAINED", action="store_true", help = "Use pretrained? True/False")

    args = parser.parse_args() 

    if not os.path.isdir(args.OUTDIR):
        os.mkdir(args.OUTDIR)
        
    experiment = f'seed_{args.RANDOMSEED}+stepsize_{args.STEP_SIZE}+groups_{args.GROUP_CNAME}_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}_transformation_{args.TRANSFORM}'
    
    logging.basicConfig(filename=f'/scratch/paa9751/mlhc-project/{args.OUTDIR}/train_log_{experiment}.log',filemode='a',level=logging.DEBUG)
    
    logging.info("********************** START **********************")
    
    if args.MODEL_TYPE == 'resnet18':
        logging.info("using resnet18")
        model = Resnet18(num_classes=1, pretrained=args.PRETRAINED)
    elif args.MODEL_TYPE =='resnet50':
        logging.info("using resnet50")
        model = Resnet50(num_classes=1, pretrained=args.PRETRAINED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    logging.info(f'device : {device}')

    # chexpert_dir = '/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_full.csv'
    # mimic_dir = '/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_full.csv'

    # chexpert_dir = '/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_middle_removed.csv'
    # mimic_dir = '/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_middle_removed.csv'

    if args.GROUP_CNAME == 'true_group_idx':
        chexpert_dir = '/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_full.csv'
        mimic_dir = '/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_full.csv'
    else:
        chexpert_dir = '/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_new_groups.csv' #removing low accuracy subgroup
        mimic_dir = '/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_new_groups.csv'
    
    torch.manual_seed(args.RANDOMSEED)
    torch.cuda.manual_seed(args.RANDOMSEED) 
    
    train_dataloader, val_dataloader, test_dataloader = load_resized_data(chexpert_dir, mimic_dir, args.IMAGE_SIZE, args.NORMALISE, args.TRANSFORM, args.NWORKERS, args.BATCH_SIZE, args.BORDER_SZ, args.GROUP_CNAME)  

    # start a new wandb run to track this script
    wandb.init(
        name = experiment,
        # set the wandb project where this run will be logged
        project=args.PROJECT_NAME,
        config={
        "learning_rate": args.LEARNING_RATE,
        "batch_size":args.BATCH_SIZE, 
        "max_epochs": args.MAX_EPOCHS, 
        "weight_decay": args.WEIGHT_DECAY, 
        "model_type":args.MODEL_TYPE, 
        "pretrained":args.PRETRAINED,
        "transform":args.TRANSFORM, 
        "normalize":args.NORMALISE,
        "group_cname":args.GROUP_CNAME,
        "step_size":args.STEP_SIZE
        }
    )
    
    #start_training_robust(args, model, train_dataloader, val_dataloader, device)
    logging.info("TRAINING DONE.")

    print('starting training!')
    model_minloss_path, model_maxacc_path,model_snsmaxacc_path = start_training_robust(args, model, train_dataloader, val_dataloader, device)

    print('predictions: ')
    
    df_minloss = get_full_data_predictions(model_minloss_path,device)
    df_maxacc = get_full_data_predictions(model_maxacc_path,device)
    df_snsmaxacc = get_full_data_predictions(model_snsmaxacc_path,device)

    df_minloss.to_csv(f'stage2-outdir-newgroups/fullpreds/{experiment}_minloss_preds.csv')
    df_maxacc.to_csv(f'stage2-outdir-newgroups/fullpreds/{experiment}_maxacc_preds.csv')
    df_snsmaxacc.to_csv(f'stage2-outdir-newgroups/fullpreds/{experiment}_snsmaxacc_preds.csv')
    
    #todo: add predictions on test data here 
    logging.info("TRAINING DONE.")

