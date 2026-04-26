# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import os, csv, time, zipfile, io, argparse, copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import progress_bar
import resnet
from model_pruning import prune_resnet50_filters
from ptflops import get_model_complexity_info
from scipy.io import savemat

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true')
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--size', default=224, type=int)
parser.add_argument('--n_epochs', default=75, type=int)
parser.add_argument('--net', default='res50')
parser.add_argument('--dataset', default='imagenet')
args = parser.parse_args()

# ---------------- SETUP ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = not args.noamp
bs = args.bs
imsize = args.size

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
best_acc = 0.0

os.makedirs("checkpoint", exist_ok=True)
os.makedirs("log", exist_ok=True)

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# =========================================================
# DATASET CLASS (UNCHANGED – AS REQUESTED)
# =========================================================
class ImageNetZipDataset(Dataset):
    def __init__(self, zip_path, split, transform=None, class_to_idx=None, val_label_map=None):
        self.zip_path = zip_path
        self.transform = transform
        self.split = split
        self.zf = None
        self.samples = []
        self.class_to_idx = class_to_idx

        prefix = f"ILSVRC/Data/CLS-LOC/{split}/"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_files = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".JPEG")]
            for name in all_files:
                if split == "train":
                    label_name = name.split("/")[-2]
                else:
                    img_id = name.split("/")[-1].replace(".JPEG", "")
                    label_name = val_label_map.get(img_id)
                if label_name and label_name in self.class_to_idx:
                    self.samples.append((name, label_name))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split}")

    def _get_zip(self):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')

    def __getitem__(self, idx):
        self._get_zip()
        img_name, label_name = self.samples[idx]
        label = self.class_to_idx[label_name]
        img_bytes = self.zf.read(img_name)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

# =========================================================
# METRICS
# =========================================================
def calculate_accuracy(model, loader):
    model.eval()
    t1, t5, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, p1 = out.max(1)
            t1 += p1.eq(y).sum().item()
            _, p5 = out.topk(5, dim=1)
            t5 += sum([y[i] in p5[i] for i in range(len(y))])
            total += y.size(0)
    return 100*t1/total, 100*t5/total

# =========================================================
# KD TRAINING
# =========================================================
def train_kd(epoch, loader, teacher, student, T=4.0, alpha=0.5):
    student.train()
    teacher.eval()
    loss_sum, correct, total = 0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                t_out = teacher(x)
            s_out = student(x)

            loss_ce = F.cross_entropy(s_out, y)
            log_s = F.log_softmax(s_out / T, dim=1)
            p_t = F.softmax(t_out / T, dim=1)
            loss_kd = F.kl_div(log_s, p_t, reduction='batchmean') * T * T
            loss = alpha * loss_ce + (1 - alpha) * loss_kd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        _, pred = s_out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

        progress_bar(batch_idx, len(loader),
            'Loss: %.3f | Acc: %.3f%%'
            % (loss_sum/(batch_idx+1), 100.*correct/total))

    return loss_sum / (batch_idx+1)

# =========================================================
# TEST
# =========================================================
def test(epoch, loader):
    global best_acc
    net.eval()
    loss_sum, t1, t5, total = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = criterion(out, y)
            loss_sum += loss.item()

            _, p1 = out.max(1)
            t1 += p1.eq(y).sum().item()
            _, p5 = out.topk(5, dim=1)
            t5 += sum([y[i] in p5[i] for i in range(len(y))])
            total += y.size(0)

            progress_bar(batch_idx, len(loader),
                'Loss: %.3f | Top1: %.2f%% | Top5: %.2f%%'
                % (loss_sum/(batch_idx+1), 100*t1/total, 100*t5/total))

    acc1, acc5 = 100*t1/total, 100*t5/total

    if acc1 > best_acc:
        best_acc = acc1
        torch.save({
            "net": net.state_dict(),
            "acc": acc1,
            "epoch": epoch
        }, "checkpoint/ETF/best_imagenet_res50.pth")

    return loss_sum/(batch_idx+1), acc1, acc5

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    zip_path = "imagenet-object-localization-challenge.zip"
    assert os.path.isfile(zip_path)

    # -------- CLASS MAP --------
    with zipfile.ZipFile(zip_path, 'r') as z:
        train_prefix = "ILSVRC/Data/CLS-LOC/train/"
        classes = sorted({n.split("/")[4] for n in z.namelist() if n.startswith(train_prefix)})
        class_to_idx = {c:i for i,c in enumerate(classes)}

        val_label_map = {}
        with z.open("LOC_val_solution.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f))
            next(reader)
            for r in reader:
                val_label_map[r[0]] = r[1].split()[0]

    # -------- DATA --------
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tf_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = ImageNetZipDataset(zip_path, "train", tf_train, class_to_idx)
    testset  = ImageNetZipDataset(zip_path, "val", tf_test, class_to_idx, val_label_map)

    trainloader = DataLoader(trainset, bs, True, num_workers=8,
                             pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader  = DataLoader(testset, 128, False, num_workers=8,
                             pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # -------- MODEL --------
    net = resnet.resnet_50(compress_rate=[0.0]*27).to(device)
    net.load_state_dict(torch.load("resnet50-19c8e357.pth", map_location='cpu', weights_only=False))

    teacher = copy.deepcopy(net).to(device)
    for p in teacher.parameters(): p.requires_grad = False

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # -------- VERIFY --------
    t1, t5 = calculate_accuracy(net, testloader)
    print(f"Teacher Top1: {t1:.2f}% | Top5: {t5:.2f}%")
    
    # Pruning logic
    input_size = (3, imsize, imsize)
    flops, params = get_model_complexity_info(net, input_size, as_strings=True)
    print("Before Pruning - FLOPs:", flops, "Params:", params)

    prune_per = torch.tensor([0.0] * 1 + [0.35] * 16 + [0.5] * 16)
    #prune_resnet50_filters(net, prune_per)

    flops_a, params_a = get_model_complexity_info(net, input_size, as_strings=True)
    print("After Pruning - FLOPs:", flops_a, "Params:", params_a)

    # -------- TRAIN --------
    log_csv = f"log/ETF_log_{args.net}_{args.dataset}.csv"
    with open(log_csv, "w") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","top1","top5"])

    list_loss, list_acc1, list_acc5 = [], [], []

    for epoch in range(args.n_epochs):
        start = time.time()
        tr_loss = train_kd(epoch, trainloader, teacher, net)
        val_loss, acc1, acc5 = test(epoch, testloader)
        scheduler.step()

        list_loss.append(val_loss)
        list_acc1.append(acc1)
        list_acc5.append(acc5)

        with open(log_csv, "a") as f:
            csv.writer(f).writerow([epoch, tr_loss, val_loss, acc1, acc5])

        savemat("ETF_results_loss_acc_epochs_imgenet.mat",
                {"list_loss": list_loss, "list_acc1": list_acc1, "list_acc5": list_acc5})

        print(f"[Epoch {epoch}] Time {time.time()-start:.1f}s | Top1 {acc1:.2f}%")

