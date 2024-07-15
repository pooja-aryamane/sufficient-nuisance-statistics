import pandas as pd
import argparse
import numpy as np
import random 
#from sklearn.linear_model import LogisticRegression
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

def start_training(args, model, train_dataloader, val_dataloader, device):
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
    sigmoid = torch.nn.Sigmoid()
    
   # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    training_losses=[]
    validation_losses=[]
    validation_accuracy=[]
    training_accuracy=[]

    train_group_losses_df = pd.DataFrame({'epoch':[]})
    val_group_losses_df = pd.DataFrame({'epoch':[]})

    auroc_metric = BinaryAUROC(thresholds=None)
    f1_metric = BinaryF1Score()
    
    #overall_unique_groups = torch.tensor([0,1,2,3]) #hardcoded 

    min_val_loss = float('inf') 
    max_val_acc = -float('inf')

    print("training started")
        
    for epoch in range(args.MAX_EPOCHS): 
        
        ########################## TRAIN ###################################

        model.train()
        train_loss=0
        correct=0
        total=0

        train_losses_by_group = {i: [] for i in range(4)}
        train_loss_by_group = {i: 0 for i in range(4)}

        train_preds_by_group = {i: [] for i in range(4)}
        train_target_by_group = {i: [] for i in range(4)}
        train_auroc_by_group = {i: [] for i in range(4)}

        train_predictions = pd.DataFrame({'group':[],'predictions':[],'preds':[],'gt':[]})
        val_predictions = pd.DataFrame({'group':[],'predictions':[],'preds':[],'gt':[]})
        
        train_group=[]
        train_predicted=[]
        train_gt=[]
        train_preds=[]
        train_path=[]

        all_preds=[]
        all_labels=[]

        #unique_groups=torch.arange(4) #only looking across 4 groups here

        for path, batch_x, batch_y,l in train_dataloader: #16, 100, 512, 512
            
            #print(len(train_dataloader))
            #print(len(l))
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x.float())
            pred = pred.squeeze(1)

            #print(batch_x)

            print('PREDICTED PROB')
        
            print(sigmoid(pred))

            print('TARGET ')
            print(batch_y)

            #print(pred)

            predicted = (sigmoid(pred)>0.5).float()
            correct += (predicted == batch_y).sum().item() 
            total+=batch_y.size(0)

            

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

            unique_groups = torch.unique(l)
            
            for g in unique_groups:
                mask = l == g
                #print(g)
                
                pred_split = pred[mask]
                target_split = batch_y[mask]
                
                #print(pred_split)
                #print(target_split)
                
                predicted_split = (sigmoid(pred_split)>0.5).float()
                
                loss = criterion(pred_split, target_split.float()) 
                
                train_losses_by_group[g.item()].append(loss.item())
                #train_preds_by_group[g.item()].extend(pred_split.cpu().tolist())
                #train_target_by_group[g.item()].extend(target_split.cpu().tolist())
                
            
            train_group.extend(l.tolist())
            train_predicted.extend(predicted.tolist())
            train_gt.extend(batch_y.tolist())
            train_preds.extend(pred.tolist())
            train_path.extend(path)


            loss = criterion(pred, batch_y.float()) 
            print('LOSS ')
            print(loss)
            train_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            #print('iteration done')
            
        #print(f'epoch {epoch}, train done')

        train_acc = correct / total
        training_accuracy.append(train_acc)   
        print(train_loss)
        train_loss = train_loss/len(train_dataloader)
        print(train_loss)
        all_preds= torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        train_auroc = auroc_metric(all_preds, all_labels)
        
        for g in unique_groups:
            #train_preds_by_group[g.item()]= torch.tensor(train_preds_by_group[g.item()])
            #train_target_by_group[g.item()] = torch.tensor(train_target_by_group[g.item()]) 
            train_loss_by_group[g.item()] = np.mean(train_losses_by_group[g.item()])
            print(f'train loss = {epoch}, group {g.item()} - ', train_loss_by_group[g.item()])
            train_group_losses_df.loc[epoch, str(g.item())] = train_loss_by_group[g.item()]
            train_group_losses_df.loc[epoch, 'epoch'] = epoch
            #train_auroc_by_group[g.item()]=auroc_metric(train_preds_by_group[g.item()], train_target_by_group[g.item()])

        training_losses.append(train_loss)

        train_predictions['group']=train_group
        train_predictions['predictions']=train_predicted
        train_predictions['gt']=train_gt
        train_predictions['preds']=train_preds
        train_predictions['train_path']=train_path

        train_group_accuracies = train_predictions[['group','predictions','gt','preds']].groupby('group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))

        ########################## VALIDATION ###################################
        
        val_loss=0
        correct=0
        total=0
        all_preds=[]
        all_labels=[]

        val_losses_by_group = {i: [] for i in range(4)} #ngroups is always 4 - this is only across true groups 
        val_loss_by_group = {i: 0 for i in range(4)}
        val_auroc_by_group = {i: [] for i in range(4)}
        val_f1_by_group = {i: [] for i in range(4)}
        
        preds_by_group = {i: [] for i in range(4)}
        target_by_group = {i: [] for i in range(4)}

        val_group=[]
        val_predicted=[]
        val_preds=[]
        val_gt=[]
        val_path = []

        print('val loop')
        
        with torch.no_grad():
            model.eval()
            
            for path, batch_x, batch_y,l in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x.float()).squeeze(1)

                predicted = (sigmoid(pred)>0.5).float()
                correct += (predicted == batch_y).sum().item() 
                total+=batch_y.size(0)
                loss = criterion(pred, batch_y.float()) 

                val_loss += loss.item()
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

                unique_groups = torch.unique(l)

                for g in unique_groups:
                    mask = l == g
                    pred_split = pred[mask]
                    target_split = batch_y[mask]
                    predicted_split = (sigmoid(pred_split)>0.5).float()
                    #preds_by_group[g.item()].extend(pred_split.cpu().tolist())
                    #target_by_group[g.item()].extend(target_split.cpu().tolist())
                    loss = criterion(pred_split, target_split.float()) 
                    val_losses_by_group[g.item()].append(loss.item())
                

                val_group.extend(l.tolist())
                val_predicted.extend(predicted.tolist())
                val_gt.extend(batch_y.tolist())
                val_preds.extend(pred.tolist())
                val_path.extend(path)

            
            for g in unique_groups:
                #preds_by_group[g.item()]= torch.tensor(preds_by_group[g.item()])
                #target_by_group[g.item()] = torch.tensor(target_by_group[g.item()]) 

                #val_f1_by_group[g.item()]=f1_metric(preds_by_group[g.item()], target_by_group[g.item()])
                #val_auroc_by_group[g.item()]=auroc_metric(preds_by_group[g.item()], target_by_group[g.item()])
                
                val_loss_by_group[g.item()] = np.mean(val_losses_by_group[g.item()])
                #print(f'val loss = {epoch} group {g.item()} - ', val_loss_by_group[g.item()])
                
                val_group_losses_df.loc[epoch, str(g.item())] = val_loss_by_group[g.item()]
                val_group_losses_df.loc[epoch, 'epoch'] = epoch
        print(val_loss)
        val_loss = val_loss / len(val_dataloader)
        print(val_loss)
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

        val_group_accuracies = val_predictions[['group','predictions','gt','preds']].groupby('group').apply(lambda group: (group['predictions'] == group['gt']).sum() / len(group))
        worst_group = val_group_accuracies.idxmin()
        worst_group_accuracy = val_group_accuracies[worst_group]

        ########################## LOGGING ###################################
        
        #if (epoch+1)%10 == 0: 
        logging.info(f"Epoch {epoch+1} : Training loss = {train_loss}, Val loss = {val_loss}") #Val Acc = {val_acc}, Val AUROC = {val_auroc}, Val F1 = {val_f1}")
        
        # wandb.log({"overall_training_loss": train_loss, "overall_validation_loss": val_loss,
        #     "overall_validation_auroc":val_auroc,"overall_validation_f1":val_f1,"overall_training_accuracy":train_acc, "overall_validation_accuracy":val_acc,"overall_training_auroc":train_auroc, "worst_group_acc":worst_group_accuracy})
        
        # for g in unique_groups:
        #     wandb.log({f"group{g}_training_accuracies":train_group_accuracies[g.item()],
        #                f"group{g}_validation_acc":val_group_accuracies[g.item()], 
        #                f"group{g}_validation_loss":val_loss_by_group[g.item()], 
        #                f"group{g}_training_loss":train_loss_by_group[g.item()]})
        
        experiment_name = f'epoch_{epoch+1}+weightdecay_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}+transform_{args.TRANSFORM}'

        experiment_name_no_epoch = f'weightdecay_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}+transform_{args.TRANSFORM}'

        if val_loss<min_val_loss:
            min_val_loss = val_loss
            logging.info(f'{epoch} - minimum val loss! saving weights...')
            torch.save(model, f'{args.OUTDIR}/lowest_val_loss.pth')
            train_predictions['epoch']=epoch
            val_predictions['epoch']=epoch
            train_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_train_preds_minloss.csv')
            val_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_val_preds_minloss.csv')

        if val_acc>max_val_acc:
            max_val_acc = val_acc
            logging.info(f'{epoch} - maximum val acc! saving weights...')
            torch.save(model, f'{args.OUTDIR}/{experiment_name_no_epoch}_max_val_acc.pth')
            train_predictions['epoch']=epoch
            val_predictions['epoch']=epoch
            train_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_train_preds_maxacc.csv')
            val_predictions.to_csv(f'{args.OUTDIR}/{experiment_name_no_epoch}_val_preds_maxacc.csv')

        if epoch%10==0:
            logging.info('saving weights...')
            torch.save(model, f'{args.OUTDIR}/models/{epoch}_model_weights.pth')
            #train_predictions.to_csv(f'{args.OUTDIR}/train_preds/{epoch}_train_preds.csv')
            #val_predictions.to_csv(f'{args.OUTDIR}/val_preds/{epoch}_val_preds.csv')
       
    np.save(f'{args.OUTDIR}/losses/train_loss_{experiment_name}.npy', np.array(training_losses))
    np.save(f'{args.OUTDIR}/losses/val_loss_{experiment_name}.npy', np.array(validation_losses))
    np.save(f'{args.OUTDIR}/accuracies/val_accuracy_{experiment_name}.npy', np.array(validation_accuracy))
    np.save(f'{args.OUTDIR}/accuracies/val_accuracy_{experiment_name}.npy', np.array(training_accuracy))

    val_group_losses_df.to_csv(f'{args.OUTDIR}/val_group_losses.csv')
    train_group_losses_df.to_csv(f'{args.OUTDIR}/train_group_losses.csv')
    
    print('done.')
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Project name 
    parser.add_argument("--PROJECT_NAME", type=str, default = 'stage-1-training', help="project_name")
 
    # Dataloader Arguments
    parser.add_argument("--M_LABEL", type=str, default = 'negbio', help="Labels for mimic: negbio or chexpert")
    parser.add_argument("--IMAGE_SIZE", type=int, default = 256, help="Image resize value")
    parser.add_argument("--TRANSFORM", action="store_true", help = 'Apply transformation? true/false')
    parser.add_argument("--NORMALISE", action="store_true", help='Apply Imagenet normalisation')
    parser.add_argument("--NWORKERS", type=int, default = 8, help="Num Workers")
    parser.add_argument("--BATCH_SIZE", type=int, default = 8, help="Num Workers")
    #parser.add_argument("--TRAIN_SPLIT", type=float, default=0.7, help = "Data train split")
    parser.add_argument("--BORDER_SZ", type=int, default=25, help="Border Size for nuisance")
    parser.add_argument("--K", type=int, default = 10000, help="Sample size from each dataset")
    parser.add_argument("--SPLIT_IDX", type=int, default = 1, help="split index")
    
    # Model Arguments
    parser.add_argument("--MAX_EPOCHS", type=int, default=200, help = "Max training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default = 1e-3, help="Learning rate")
    parser.add_argument("--WEIGHT_DECAY", type=float, default = 1e-3, help="Weight Decay")
    #parser.add_argument("--LOGFILE", type=str, default="./logs", help = "Dir of log file")
    parser.add_argument("--OUTDIR", type=str, default="./out", help = "Dir of out files - losses/weights")
    parser.add_argument("--MODEL_TYPE", type=str, default="resnet18", help = "resnet18 or resnet50")
    parser.add_argument("--PRETRAINED", action="store_true", help = "Use pretrained? True/False")

    args = parser.parse_args() 

    if not os.path.isdir(args.OUTDIR):
        os.mkdir(args.OUTDIR)
    
    experiment = f'weightdecay_{str(args.WEIGHT_DECAY)}+pretrained_{str(args.PRETRAINED)}+batchsize_{str(args.BATCH_SIZE)}+lr_{args.LEARNING_RATE}'

    logging.basicConfig(filename=f'/scratch/paa9751/mlhc-project/{args.OUTDIR}/training_log.log',filemode='a',level=logging.DEBUG)
    
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

    chexpert_dir = '/scratch/paa9751/mlhc-project/resized_data/chexpert'
    mimic_dir = '/scratch/paa9751/mlhc-project/resized_data/mimic'
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
  
    train_dataloader, val_dataloader, test_dataloader = load_resized_data(chexpert_dir, mimic_dir, args.IMAGE_SIZE, args.NORMALISE, args.TRANSFORM, args.NWORKERS, args.BATCH_SIZE, args.BORDER_SZ, 'true_group_idx', args.SPLIT_IDX) 
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=args.PROJECT_NAME,

    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": args.LEARNING_RATE,
    #     "batch_size":args.BATCH_SIZE, 
    #     "max_epochs": args.MAX_EPOCHS, 
    #     "weight_decay": args.WEIGHT_DECAY, 
    #     "model_type":args.MODEL_TYPE, 
    #     "pretrained":args.PRETRAINED,
    #     "transform":args.TRANSFORM, 
    #     "normalize":args.NORMALISE,
    #     "border_size":args.BORDER_SZ, 
    #     "split_index": args.SPLIT_IDX
    #     }
    # )
    
    print('starting training!')
    start_training(args, model, train_dataloader, val_dataloader, device)
    
    logging.info("TRAINING DONE.")

