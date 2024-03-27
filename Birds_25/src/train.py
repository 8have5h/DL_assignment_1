
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from resnet import ResNet
import json
from collections import defaultdict
from time import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
def train(args, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    net = ResNet(norm_type = args['norm_type'])
    print(net)
    net = net.to(device)
    print("Model Loaded on Device:", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    schedulers = [
        optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], verbose=False)
        ]
    scheduler =  schedulers[1] #Check for self.epochs param
    
    loss_tracker = defaultdict(list)
    accuracy_tracker = defaultdict(list)    
    time_tracker = defaultdict(list)
    f1_macro_score_tracker = defaultdict(list)
    f1_micro_score_tracker = defaultdict(list)


    best_accuracy = -1
    best_accu_epoch = -1

    print("\n\n---------------------------- MODEL TRAINING BEGINS ----------------------------")
        
    t0 = time()
    for epoch in range(args['epochs']):
        print("\n#------------------ Epoch: %d ------------------#" % epoch)

        train_loss = []
        correct_pred = 0
        total_samples = 0
        orig_labels = []
        preds = []
        
        net.train()
        for idx, batch in enumerate(train_loader):
            # print(idx, len(batch[0]))
            optimizer.zero_grad()
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            
            loss = criterion(outputs, labels)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)
            orig_labels.extend(labels.squeeze().tolist())
            preds.extend(pred.squeeze().tolist())
            # print("Batch: {}, Loss: {}, Accuracy: {}%".format(idx, loss.item(), round(correct_pred/total_samples*100, 2) ))

        loss_tracker["train"].append(np.mean(train_loss))
        accuracy_tracker["train"].append(round(correct_pred/total_samples*100, 2))
        f1_macro_score_tracker["train"].append(round(f1_score(orig_labels, preds, average='macro'), 4))
        f1_micro_score_tracker["train"].append(round(f1_score(orig_labels, preds, average='micro'), 4))


        scheduler.step()
        print("validating...")
        net.eval()
        correct_pred = 0
        total_samples = 0
        val_loss = []
        preds = []
        orig_labels = []
        for idx, batch in enumerate(val_loader):
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)        
            val_loss.append(loss.item())

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)
            orig_labels.extend(labels.squeeze().tolist())
            preds.extend(pred.squeeze().tolist())

        
        loss_tracker["val"].append(np.mean(val_loss))
        val_accuracy = round(correct_pred/total_samples*100, 2)
        accuracy_tracker["val"].append(val_accuracy)
        f1_macro_score_tracker["val"].append(round(f1_score(orig_labels, preds, average='macro'), 4))
        f1_micro_score_tracker["val"].append(round(f1_score(orig_labels, preds, average='micro'), 4))


        t1 = time()

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Train Accuracy: {}%, Validation Loss: {}, Validation Accuracy: {}%, Train F1 Macro: {}, Train F1 Micro: {}, Val F1 Macro: {}, Val F1 Micro: {}".format(epoch, round((t1-t0)/60,2), loss_tracker["train"][-1], accuracy_tracker["train"][-1], loss_tracker["val"][-1], accuracy_tracker["val"][-1], f1_macro_score_tracker["train"][-1], f1_micro_score_tracker["train"][-1], f1_macro_score_tracker["val"][-1], f1_micro_score_tracker["val"][-1]))
        time_tracker['train'].append(round((t1-t0)/60,2))
        model_state = {
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }

        print("Epoch: {}, Saving Model Checkpoint: {}".format(epoch, now.strftime("%d-%m-%y %H:%M")))
        
        torch.save(net, os.path.join(args['checkpoint_dir'] , "latest_checkpoint_{}.pth".format(args['norm_type'])))
        with open(os.path.join(args['checkpoint_dir'], "training_progress_{}.json".format(args['norm_type'])), "w") as outfile:
            json.dump(model_state, outfile)
        
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_accu_epoch = epoch

            model_state = {
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }
            
            print("Best Validation Accuracy Updated = {}%, Last Best = {}%".format(val_accuracy, best_accuracy))
            print("Saving Best Model Checkpoint:", now.strftime("%d-%m-%y %H:%M"))

            torch.save(net, os.path.join(args['checkpoint_dir'] , "best_val_checkpoint_{}.pth".format(args['norm_type'])))
            with open(os.path.join(args['checkpoint_dir'] , "training_progress_{}.json".format(args['norm_type'])), "w") as outfile:
                json.dump(model_state, outfile)


        with open(os.path.join(args['result_dir'],"loss_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        with open(os.path.join(args['result_dir'],"accuracy_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(accuracy_tracker, outfile)

        with open(os.path.join(args['result_dir'],"time_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(time_tracker, outfile)

        with open(os.path.join(args['result_dir'],"f1_macro_score_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(f1_macro_score_tracker, outfile)

        with open(os.path.join(args['result_dir'],"f1_micro_score_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(f1_micro_score_tracker, outfile)
    return


