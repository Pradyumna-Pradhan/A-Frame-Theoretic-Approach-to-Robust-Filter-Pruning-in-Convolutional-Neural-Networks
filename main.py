# -*- coding: utf-8 -*-
'''
Filter Pruning of VGG-16 network Training on the CIFAR-10 using frame theoretic approach
Authors: Pradyumna Pradhan, Pradip Sasmal, and Ramunaidu Randhi
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import copy

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models import *
from utils import progress_bar
from model_pruning import our_pruned_model
from model_pruning import prune_one_layer
from model_pruning import our_pruned_model_ae_cbmi
from model_pruning import our_pruned_model_ae
from ptflops import get_model_complexity_info
from randomaug import RandAugment
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset
from scipy.io import savemat


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vgg')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='128')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='300')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')

args = parser.parse_args()

# take in args
usewandb = args.nowandb
if usewandb:
    import wandb

    watermark = "{}_lr{}_{}".format(args.net, args.lr, args.dataset)
    wandb.init(project="cifar-challenge",
               name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize

# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
    dataset_class = torchvision.datasets.CIFAR100
else:
    raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

# Set up class names based on the dataset
if args.dataset == 'cifar10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    # CIFAR100 has 100 classes, so we don't list them all here
    classes = None

# Model factory..
# print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18(num_classes=num_classes)
elif args.net == 'vgg':
    net = VGG()
    # net = VGG('VGG16')
    # net = VGG('VGG19', num_classes=num_classes)
elif args.net == 'res34':
    net = ResNet34(num_classes=num_classes)
elif args.net == 'res50':
    net = ResNet50(num_classes=num_classes)
elif args.net == 'res101':
    net = ResNet101(num_classes=num_classes)
elif args.net == "convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net == "mlpmixer":
    from models.mlpmixer import MLPMixer

    net = MLPMixer(
        image_size=32,
        channels=3,
        patch_size=args.patch,
        dim=512,
        depth=6,
        num_classes=num_classes
    )
elif args.net == "vit_small":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_tiny":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=4,
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "simplevit":
    from models.simplevit import SimpleViT

    net = SimpleViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512
    )
elif args.net == "vit":
    # ViT for cifar10/100
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "dyt":
    # DyT for cifar10/100
    net = DyT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.net == "cait":
    from models.cait import CaiT

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "cait_small":
    from models.cait import CaiT

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "swin":
    from models.swin import swin_t

    net = swin_t(window_size=args.patch,
                 num_classes=num_classes,
                 downscaling_factors=(2, 2, 2, 1))
elif args.net == "mobilevit":
    net = mobilevit_xxs(size, num_classes)
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# For Multi-GPU
if 'cuda' in device:
    # print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net)  # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)


def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1)


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ETF/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    log_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.txt'
    with open(log_file, 'a') as appender:
        appender.write(content + "\n")
    return test_loss / (batch_idx + 1), acc


list_loss = []
list_acc = []


def calculate_accuracy(model, loader, device):
    model.to(device)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1 predictions
            # _, top1_preds = torch.max(outputs, dim=1)
            # top1_correct += (top1_preds == labels).sum().item()
            _, top1_preds = outputs.max(1)
            top1_correct += top1_preds.eq(labels).sum().item()

            # Top-5 predictions
            _, top5_preds = outputs.topk(5, dim=1)
            top5_correct += sum([1 if labels[i] in top5_preds[i] else 0 for i in range(len(labels))])

            total += labels.size(0)

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    return top1_acc, top5_acc


# Knowledge Distillation training
def train_kd(epoch, trainloader, teacher, student,
             T=4.0, alpha=0.5):
    """
    T      : temperature
    alpha  : weight for CE vs KD
    loss = alpha * CE(student, y) + (1-alpha) * T^2 * KL(student_T || teacher_T)
    """
    print('\n[KD] Epoch: %d' % epoch)
    teacher.eval()
    student.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            # teacher forward (no grad)
            with torch.no_grad():
                logits_t = teacher(inputs)

            # student forward
            logits_s = student(inputs)

            # hard-label loss (normal CE)
            loss_ce = F.cross_entropy(logits_s, targets)

            # soft-label KD loss
            log_p_s = F.log_softmax(logits_s / T, dim=1)
            p_t = F.softmax(logits_t / T, dim=1)
            loss_kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = logits_s.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (batch_idx + 1)



if usewandb:
    wandb.watch(net)


if __name__ == '__main__':
    # Data Set generation and loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2;
        M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    net.cuda()
    # For Loading the trained model
    checkpoint = torch.load('./cifar10_vgg_16_bn_93_96.pt', map_location='cpu')
    # # 3️⃣ Load saved weights
    net.load_state_dict(checkpoint['state_dict'])

    # 4️⃣ Set model to evaluation mode (important for inference)
    net.eval()

    print("✅ Model loaded successfully!")

    # ---- NEW: create TEACHER (unpruned) ----
    teacher = copy.deepcopy(net).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False  # freeze teacher

    input_size = (3, 32, 32)
    flops, params = get_model_complexity_info(net, input_size, as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs before pruning: {flops}")
    print(f"Parameters before pruning: {params}")

    print("Pruning started")
    #prune_per = torch.tensor([0.55] * 2 + [0.65]*2 + [0.80]*3 +[0.60]*4 + [0.65] * 2)
    prune_per = torch.tensor([0.25] * 5 + [0.65]*8 ) #Pruning percentage
    our_pruned_model_ae(net, prune_per)  # Auto-Encoder based filter Pruning
    #our_pruned_model(net, prune_per)
    # # our_pruned_model_ae_cbmi(net, prune_per)
    flops_a, params_a = get_model_complexity_info(net, input_size, as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs after pruning: {flops_a}")
    print(f"Parameters after pruning: {params_a}")

    print("Training the pruned model with Knowledge Distillation")
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()

        # KD training: teacher (fixed) -> student (pruned net)
        trainloss = train_kd(epoch, trainloader, teacher=teacher, student=net,
                             T=4.0, alpha=0.5)

        val_loss, acc = test(epoch)

        scheduler.step(epoch - 1)

        list_loss.append(val_loss)
        list_acc.append(acc)

        if usewandb:
            wandb.log({'epoch': epoch,
                       'train_loss': trainloss,
                       'val_loss': val_loss,
                       "val_acc": acc,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time() - start})

        csv_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)
        print(list_acc)

        # save the loss and accuracy w.r.t. epoch
        savemat("results_loss_acc_epochs.mat", {
            "list_acc": list_acc,
            "list_loss": list_loss
        })


    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}_{}.h5".format(args.net, args.dataset))

#########################################################################
##########################################################