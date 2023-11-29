

from collections import OrderedDict
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import math

import time
import torchvision
from torchvision import datasets, transforms
from PIL import Image

import torchvision.models as models
import wandb
import logging
from torch.optim.lr_scheduler import LambdaLR

from torch.cuda import amp
from torch.nn import functional as F

def finetune(args, finetune_dataset, test_loader, model, criterion):
    logger = logging.getLogger(__name__)
    logger = logging.getLogger()
    #model.drop = nn.Identity()
    #train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        finetune_dataset,
        batch_size=args.finetune_batch_size,
        shuffle = True)
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay,
                          nesterov=True)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast():
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:

            test_loss, top1 = evaluate(args, test_loader, model, criterion)

            
            """ wandb.log({"test/loss": test_loss,
                        "test/acc@1": top1} ) """

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            """ args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch) """
            """ wandb.log({"finetune/train_loss": losses.avg,
                    "finetune/test_loss": test_loss,
                    "finetune/acc@1": top1,
                    "finetune/acc@5": top5}) """


            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        """ if args.local_rank in [-1, 0]:
            #args.writer.add_scalar("result/finetune_acc@1", args.best_top1)
            wandb.log({"result/finetune_acc@1": args.best_top1}) """
    return

def create_loss_fn(args):
    # if args.label_smoothing > 0:
    #     criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    # else:  
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.to(args.device)


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))

    target = target.to(torch.device('cpu'))

    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)

    pred = idx.narrow(1, 0, maxk).t()

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))



    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc1 = accuracy(outputs, targets)[0]

            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"acc: {top1.avg.item():.2f}")

        test_iter.close()
        return losses.avg, top1.avg

def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    wandb.init(project="Eli")
    logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    """ if wandb.run.get_url() == '':
        logger.warning("Running *without* WandB logging.")
    else:
        logger.info(f"Logging this run to {wandb.run.get_url()}.") """

    logger.info("***** Running Training *****")
    logger.info(f"   Total steps = {args.total_steps}")


    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)


    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

    
        # error occurs ↓
        # images_l, targets = labeled_iter.next()
        try:
            images_l, targets = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            images_l, targets = next(labeled_iter)


    
        # error occurs ↓
        # (images_uw, images_us), _ = unlabeled_iter.next()
        try:
            (images_uw, images_us), _ = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            (images_uw, images_us), _ = next(unlabeled_iter)
        

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        
        images_uw = images_uw.to(args.device)
        
        images_us = images_us.to(args.device)
       
        targets = targets.to(args.device)
       
        with amp.autocast():
            batch_size = images_l.shape[0]
        
            t_images = torch.cat((images_l, images_uw, images_us))
       
            t_logits = teacher_model(t_images)
      
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits



            t_loss_l = criterion(t_logits_l, targets)


            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)

            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        """ if args.ema > 0:
            avg_student_model.update_parameters(student_model) """

        with amp.autocast():
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            #dot_product = s_loss_l_new - s_loss_l_old
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # test
            # t_loss_mpl = torch.tensor(0.).to(args.device)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()


        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()

        wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()

            wandb.log({"train/1.s_loss": s_losses.avg,
                        "train/2.t_loss": t_losses.avg,
                        "train/3.t_labeled": t_losses_l.avg,
                        "train/4.t_unlabeled": t_losses_u.avg,
                        "train/5.t_mpl": t_losses_mpl.avg,
                        "train/6.mask": mean_mask.avg})

            test_model = student_model
            test_loss, top1 = evaluate(args, test_loader, test_model, criterion)
            test_loss_t, top1_t = evaluate(args, test_loader, teacher_model, criterion)

            
            wandb.log({"test/loss": test_loss,
                        "test/acc@1": top1,
                        "test/test_loss_teacher": test_loss_t,
                        "test/acc@1 teacher": top1_t })
            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")
            logger.info(f"top-1 acc teacher: {top1_t:.2f}")


            save_checkpoint(args, {
                'step': step + 1,
                'teacher_state_dict': teacher_model.state_dict(),
                'student_state_dict': student_model.state_dict(),
                'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                'best_top1': args.best_top1,
                'teacher_optimizer': t_optimizer.state_dict(),
                'student_optimizer': s_optimizer.state_dict(),
                'teacher_scheduler': t_scheduler.state_dict(),
                'student_scheduler': s_scheduler.state_dict(),
                'teacher_scaler': t_scaler.state_dict(),
                'student_scaler': s_scaler.state_dict(),
            }, is_best)

    return



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_and_visualize(teacher, testloader, device):
    preds = []
    images = []
    labels = []
    total = 0
    correct = 0
    
    teacher.eval()
    
    with torch.no_grad():
        for img, label in iter(testloader):
            img = img.to(device)
            label = label.to(device)
            
            # Forward pass
            output = teacher(img)
            # Get predictions
            _, pred = torch.max(output.data, 1)
            # Register all wrong predictions
            wrongs = ~pred.eq(label)
            total += len(label)
            correct += (~wrongs).sum()
            
            # Check if there is a True value in wrongs
            if wrongs.any():
                labels.append(label[wrongs])
                preds.append(pred[wrongs])
                images.append(img[wrongs])
    
    labels = torch.hstack(labels)
    preds = torch.hstack(preds)
    images = torch.vstack(images)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
        
    Titles = ["Damage" if i.item() == 0 else "No Damage" for i in preds]
    show_images(images, titles=Titles)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_model(model_name, version = None, out = 2):
    if model_name == "mobile":
        if version == 2:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            nr_filters = model.classifier[1].in_features  #number of input features of last layer
            model.classifier[1] = nn.Linear(nr_filters, out)
        elif version == 3:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
            nr_filters = model.classifier[3].in_features  #number of input features of last layer
            model.classifier[3] = nn.Linear(nr_filters, out)
        else:
            raise Exception("Model not implemented")
    elif model_name == "ghost":
        model =  torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        nr_filters = model.classifier.in_features  #number of input features of last layer
        model.classifier = nn.Linear(nr_filters, out)
    elif model_name == "resnet":
        if version == 18:
            model =  models.resnet18(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        elif version == 34:
            model =  models.resnet34(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        elif version == 50:
            model =  models.resnet50(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        elif version == 101:
            model =  models.resnet101(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        elif version == 152:
            model =  models.resnet152(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        else:
            raise Exception("Model not implemented")
    elif model_name == "resnext":
        if version == 50:
            model =  models.resnext50_32x4d(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        elif version == 101:
            model =  models.resnext101_32x8d(pretrained=True)
            nr_filters = model.fc.in_features  #number of input features of last layer
            model.fc = nn.Linear(nr_filters, out)
        else:
            raise Exception("Model not implemented")
    else:
        raise Exception("Model not implemented")
    return model

class TransformMPL(object):
    def __init__(self):

        self.ori = transforms.Compose([
            transforms.Resize((224,224)),])
        self.aug = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandAugment()
            ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
    ),])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)

class UnsDataset(Dataset):
    def __init__(self, directory, images, transform=False):
        self.images = images
        self.transform = transform
        self.directory = directory
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filepath = self.directory + "/" + self.images[idx]
        image = Image.open(image_filepath)

        if self.transform is not None:
            image = self.transform(image)
        return image, image_filepath
    
def show_images(images, title = "", titles=[]):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
                if titles:
                    ax.set_title(str(titles[i]))
                    i +=1

    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

def get_data(traindir, testdir, uns_dir, uns_images):
    

    transform_labeled = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
    ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
    ),
    ])

    

    train_labeled_dataset = datasets.ImageFolder(traindir,transform=transform_labeled)
    test_dataset = datasets.ImageFolder(testdir,transform=transform_val)
    train_unlabeled_dataset = UnsDataset(uns_dir, uns_images, transform = TransformMPL())

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class Args:
    def __init__(self):
        self.best_top1 = 0
        self.name = "MLP"
        self.grad_clip = 1e9
        self.seed = 2
        self.local_rank = -1
        self.momentum = 0.9
        self.start_step = 0
        self.num_classes = 2
        #self.num_labeled = 4000
        self.total_steps = 30000
        self.eval_step = 10
        #self.eval_step = 5
        self.batch_size = 16
        self.teacher_lr = 0.05
        self.student_lr = 0.05
        self.weight_decay = 5e-4
        self.mu = 4
        self.label_smoothing = 0
        self.temperature = 0.7
        self.threshold = 0.6
        self.lambda_u = 8
        self.warmup_steps = 500
        self.uda_steps = 500
        self.student_wait_steps = 500
        self.teacher_dropout = 0.2
        self.student_dropout = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = "C:/Users/UTENTE/Desktop/defectEli/save"
        self.finetune_batch_size = 16
        self.finetune_epochs = 100
        self.finetune_lr = 3e-5
        self.finetune_momentum = 0.9
        self.finetune_weight_decay = 0
        self.amp = True
        self.world_size = 1
        self.ema = 0.999
        self.num_eval = 0
