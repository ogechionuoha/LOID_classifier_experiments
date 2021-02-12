#imports
import os
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data.sampler import SubsetRandomSampler
from helper_fns import *
from resnet_models import ResNet50_GradCam, ResNet50_CBAM


from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

torch.backends.cudnn.benchmark = True

# data preprocessing
data = '/home/ogechi/Hotels-50K/images/uk_not_uk/train/'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

begin = time.time()

tr_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
#test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

tr_dataset = datasets.ImageFolder(data, transform=tr_transform)

end = time.time() - begin
class_names = tr_dataset.classes
print(f'loading complete in {end // 60:.0f}m {end % 60:.0f}s')

tr_class_counts = np.sum(np.array(tr_dataset.targets) == 0), np.sum(np.array(tr_dataset.targets) == 1)

print(tr_class_counts)

train_count, val_count = int(0.75 * len(tr_dataset)), int(0.15 * len(tr_dataset))
test_count = len(tr_dataset) - (train_count + val_count)
train_set, val_set, test_set = torch.utils.data.random_split(tr_dataset, [train_count, val_count, test_count])
dataset_sizes = len(train_set), len(val_set), len(test_set)
print(dataset_sizes)

#loaders
batchsize = 64
workers = 2
pinmemory = True
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, num_workers=workers, shuffle=True, pin_memory=pinmemory)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batchsize, num_workers=workers, shuffle=True, pin_memory=pinmemory)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, num_workers=workers, shuffle=True, pin_memory=pinmemory)

def train(model, criterion_set, optimizer, epochs, scheduler, class_names, device = 'cpu', board_writer=None, save_folder = None):
    model = model.to(device)
    valid_loss_min = np.Inf
    start = time.time()
    for epoch in range(epochs):

        print(f'Epoch {epoch}/{epochs-1}')
        print('-'*15)

        for phase in ['train','val']:

            running_loss = 0.0
            running_correct = 0

            class_probs = []
            class_preds = []
            y_target = np.array([])

            if phase == 'train':
                model.train()  # Set model to training mode
                loader = train_loader
                dataset_size = dataset_sizes[0]
                filename = save_folder+'/train.csv'
                criterion = criterion_set[0]
            else:
                model.eval()
                loader = valid_loader
                dataset_size = dataset_sizes[1]
                filename = save_folder+'/val.csv'
                criterion = criterion_set[1]
            i = 0
            begin_phase = time.time()
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_target = np.concatenate((y_target,labels.cpu()))
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
                    class_probs.append(class_probs_batch)
                    class_preds.append(preds.cpu())

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_correct += torch.sum(preds == labels.data)

                end = time.time() - begin_phase
                i+=1
                if i%1000 == 0:
                    print(f'Phase: {phase} Batch {i} complete in {end // 60:.0f}m {end % 60:.0f}s')

            if phase == 'train':
                scheduler.step()

            epoch_probs = torch.cat([torch.stack(batch) for batch in class_probs])
            epoch_preds = torch.cat(class_preds)

            for i in range(len(class_names)):
                add_pr_curve_tensorboard(board_writer, phase, i, epoch_probs, epoch_preds, class_names, global_step=epoch)

            epoch_loss = running_loss / dataset_size
            epoch_accuracy = running_correct.double() / dataset_size

            # save model if validation loss has decreased
            if phase == 'val' and epoch_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saved updated model.'.format(valid_loss_min, epoch_loss))
                torch.save(model.state_dict(), save_folder+'/checkpoint.pt')
                valid_loss_min = epoch_loss

            log_metrics(filename, y_target, epoch_preds, epoch_loss, epoch_loss <= valid_loss_min, writer=board_writer, epoch=epoch)

            time_elapsed = time.time() - start

            print(f'{phase} Loss: {epoch_loss:.4f}; Accuracy: {epoch_accuracy:.4f}; Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

            if board_writer is not None:
                board_writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
                board_writer.add_scalar(f'{phase}/accuracy', epoch_accuracy, epoch)

        print()

    if board_writer is not None:
        board_writer.flush()
        board_writer.close()

    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model


#model settings
model_name = 'resnet50_finetuned_gradcam_cbam'
results_folder = './results/'
save_folder = results_folder + model_name
train_results = save_folder +'/train.csv'
val_results = save_folder +'/val.csv'
checkpoint_path = save_folder +'/checkpoint.pt'

logs = SummaryWriter(f'./logs/{model_name}')

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

if not os.path.isdir(results_folder + model_name):
    os.mkdir(results_folder + model_name)

if not os.path.exists(train_results):
    with open(train_results, 'a') as train_result:
        header = ",".join(['loss', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'precision_0', 'precision_1', 'recall_0', 'recall_1', 'f1_0', 'f1_1', 'count_0', 'count_1','auroc','\n'])
        train_result.write(header)

if not os.path.exists(val_results):
    with open(val_results, 'a') as val_result:
        header = ",".join(['loss', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'precision_0', 'precision_1', 'recall_0', 'recall_1', 'f1_0', 'f1_1', 'count_0', 'count_1','auroc','saved','\n'])
        val_result.write(header)


#calculate loss weights 
tr_loss_weights = 1.0/torch.Tensor([np.sum(np.array(train_set.dataset.targets)[train_set.indices] == 0) , np.sum(np.array(train_set.dataset.targets)[train_set.indices] == 1)])
valid_loss_weights = 1.0/torch.Tensor([np.sum(np.array(val_set.dataset.targets)[val_set.indices] == 0) , np.sum(np.array(val_set.dataset.targets)[val_set.indices] == 1)])
test_loss_weights = 1.0/torch.Tensor([np.sum(np.array(test_set.dataset.targets)[test_set.indices] == 0) , np.sum(np.array(test_set.dataset.targets)[test_set.indices] == 1)])
tr_loss_weights, valid_loss_weights, test_loss_weights

model= ResNet50_CBAM(num_classes=len(class_names), visualise=False, pretrained=True, finetune=True, attn_type='CBAM')

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

balance_loss = True

if balance_loss:
    print("Using scaled loss function")
    train_criterion = nn.CrossEntropyLoss(weight=tr_loss_weights.to(device))
    valid_criterion = nn.CrossEntropyLoss(weight=valid_loss_weights.to(device))
else:
    print("Using unscaled loss function")
    train_criterion = nn.CrossEntropyLoss()
    valid_criterion = nn.CrossEntropyLoss()

criterion_set = [train_criterion, valid_criterion]

print(device)

if __name__ == '__main__':
    train(model, criterion_set, optimizer, epochs= 90, scheduler=exp_lr_scheduler,class_names= class_names, device = device, board_writer=logs, save_folder = save_folder)
