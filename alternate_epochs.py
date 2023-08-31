import copy
import torch
from torch.optim import Adam
import torchvision
from torch import nn
import torchvision.transforms as transforms
from lightly.loss import DINOLoss
from lightly.data import LightlyDataset
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, activate_requires_grad
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import wandb
import torchmetrics
import argparse
import matplotlib.pyplot as plt
import numpy as np


# IMPORTANT:
# make sure to add this "T.Resize((224,224))," to dino transform file

class DINO(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        input_dim = backbone.embed_dim
        self.args = args
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 2048, 256, args.output_dim, batch_norm=(not args.no_batchnorm)
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, 2048, 256, args.output_dim, batch_norm=(not args.no_batchnorm)
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=args.output_dim, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def forward(self, views):
        teacher_out = [self.forward_teacher(view) for view in views[:2]]
        student_out = [self.forward(view) for view in views]
        return teacher_out, student_out

    def set_params(self, lr_factor, max_epochs):
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

    def configure_optimizers(self):
        param = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        if not self.args.Adam:
            optim = torch.optim.SGD(
                param,
                lr=self.args.learning_rate * self.lr_factor,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            optim = Adam(self.parameters(), lr=self.args.learning_rate)
        if self.args.no_scheduler:
            return [optim]
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
            return [optim], [cosine_scheduler]


class Classifier(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.feature_extractor = model
        self.classifier = nn.Sequential(
            nn.Linear(384, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

# Pretraining function
def pretrain(args):
    print("starting pretraining")
    wandb.init(project='transformer 224 unsup pretraining')
    # Log the arguments to wandb
    wandb.config.update(args)

    bs = args.batch_size
    num_workers = 16

    lr_factor = bs / 256
    max_epochs = args.pretrain_epochs

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    dino_transform = DINOTransform(global_crop_size=224, local_crop_size=224)
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=dino_transform)

    model = DINO(args)
    model.set_params(lr_factor, max_epochs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    for epoch in range(max_epochs):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            teacher_out, student_out = model(batch)
            loss = model.criterion(teacher_out, student_out, epoch=epoch)
            wandb.log({"pretraining_loss": loss.item()})
            loss.backward()
            optimizer.step()

        scheduler.step()

    wandb.finish()

    activate_requires_grad(model.student_backbone)
    return model.student_backbone


# Supervised training function
def supervised_train(model, args, train_loader, val_loader):
    print("starting sup training")
    wandb.init(project='transformer 224 sup training')
    # Log the arguments to wandb
    wandb.config.update(args)

    optimizer = Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    max_epochs = args.supervised_epochs

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch in val_loader:
                x, y = batch
                logits = model(x)
                loss = loss_fn(logits, y)
                preds = torch.argmax(logits, dim=1)
                total_loss += loss.item() * len(x)
                total_correct += torch.sum(preds == y).item()
                total_samples += len(x)

            val_loss = total_loss / total_samples
            val_acc = total_correct / total_samples
            wandb.log({"val_loss": val_loss, "val_acc": val_acc})

    wandb.finish()


def create_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.supervised_batch_size, shuffle=True,
                                               num_workers=12)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.supervised_batch_size, shuffle=False,
                                             num_workers=12)

    return train_loader, val_loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--supervised_epochs', type=int, default=20)
    parser.add_argument('--no_batchnorm', action='store_true')
    parser.add_argument('--Adam', action='store_true')
    parser.add_argument('--no_scheduler', action='store_true')
    parser.add_argument('--output_dim', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=6e-2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--supervised_batch_size', type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    pretrained_feature_extractor = pretrain(args)
    pretrained_model = Classifier(pretrained_feature_extractor, 10)

    train_loader, val_loader = create_datasets()

    supervised_train(pretrained_model, args, train_loader, val_loader)