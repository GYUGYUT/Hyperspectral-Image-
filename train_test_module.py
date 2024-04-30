import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm
from confusion_matrix import *
from report_save import *
from cal_accuracy import *
def train(device,args,model,train_data,val_data,loss_fn, optimizer, epoch,wandb,early_stopping):
    model.train()
    loss = 0.0
    total_loss = []
    train_acc = []
    for batch_id, (X, y) in enumerate(tqdm(train_data,"Train_epoch : %d, lr : %f, progress"% (epoch, optimizer.param_groups[0]['lr']))):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss.append(loss.cpu().item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc.append(calc_accuracy(pred, y))
    wandb.log({"train_acc": np.mean(train_acc)*100, "train_loss": np.mean(total_loss)},step=epoch)    
    print("epoch {} batch id {} loss {:.6f} train acc {:.6f}%".format(epoch, batch_id+1, np.mean(total_loss), np.mean(train_acc)*100))
    # if(epoch%10 == 0): #pt저장
    #     path = os.path.join(args.save_path, 'One_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
    #     path2 = os.path.join(args.save_path2, 'Two_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
    #     torch.save(model.state_dict(),path2)
    #     torch.save(model.module.state_dict(), path)
    
    val_acc = []
    val_loss = 0.0
    model.eval()
    val_total_loss = []
    for batch_id, (X, y) in enumerate(tqdm(val_data," %d val!!!!"% epoch)):
        example_image = []
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)
            val_loss = loss_fn(pred, y)
            val_total_loss.append(val_loss.item())

        val_acc.append(calc_accuracy(pred, y))
        example_image.append(wandb.Image(X[0],caption="Pred:{} Truth:{}".format(pred[0],y[0])))
    wandb.log({"Exampes":example_image,
                "val_acc": np.mean(val_acc)*100, 
                "val_loss": np.mean(val_total_loss)})
    print("-------------->val loss {:.6f} val_acc {:.6f}%".format(np.mean(val_total_loss),np.mean(val_acc)*100))
    early_stopping(np.mean(val_acc)*100, model,args,epoch)
    
    
    if early_stopping.early_stop:
        print("Early stopping")


def test(device,args, model, dataloader,loss_fn,wandb,early_stopping, check_GPU_STATE):

    if check_GPU_STATE == 2:
        print("GPU USE 2EA")
        print("best_model : ",early_stopping.path2)
        model.load_state_dict(torch.load(early_stopping.path2))
    else:
        print("GPU USE 1EA")
        print("best_model : ",early_stopping.path)
        model.load_state_dict(torch.load(early_stopping.path)) 

    test_acc = []
    test_loss = 0.0
    model.eval()
    test_total_loss = []

    result = []
    label_result = []

    con_y_pred = []
    con_y_label = []
 
    file_path = os.path.join(args.save_path_report, '{}_report.txt'.format(args.arch))
 
    
    for batch_id, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)

            test_loss = loss_fn(pred.to(device), y.to(device))
            test_total_loss.append(test_loss.item())
            _, outputs2 = torch.max(pred, 1)
            _, outputs3 = torch.max(y, 1)
            con_y_pred.extend(outputs2.cpu().data.numpy())
            con_y_label.extend(outputs3.cpu().data.numpy())
            result.append(outputs2.tolist())
            label_result.append(outputs3.tolist())
        test_acc.append(calc_accuracy(pred, y))
    print("-------------->test loss {:.6f} test_acc {:.6f}%".format(np.mean(test_total_loss),np.mean(test_acc)*100))
    wandb.log({"test_acc": np.mean(test_acc)*100, "test_loss": np.mean(test_total_loss)})
    confusion(con_y_label,con_y_pred,args.classes,args)
    report(file_path,con_y_label,con_y_pred,args.classes)
    print("end")
    wandb.finish()

