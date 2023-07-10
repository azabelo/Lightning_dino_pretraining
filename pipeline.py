import copy
import pytorch_lightning as pl
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
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np


# IMPORTANT:
# make sure to add this "T.Resize((224,224))," to dino transform file

class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        resnet = torchvision.models.resnet50()
        input_dim = list(resnet.children())[-1].in_features
        backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        self.student_backbone = backbone
        # self.student_head = DINOProjectionHead(
        #     input_dim, 512, 64, 2048, freeze_last_layer=1
        # )
        # self.teacher_backbone = copy.deepcopy(backbone)
        # self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        self.student_head = DINOProjectionHead(
            input_dim, 2048, 256, 2048, batch_norm=True
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, 2048, 256, 2048, batch_norm=True
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        # print("global: ", views[0][0].shape, " local: ", views[2][0].shape)
        # image = views[2][0].cpu()
        # image_np = image.numpy()
        # image_np = np.transpose(image_np, (1, 2, 0))
        # plt.imshow(image_np)
        # plt.axis('off')
        # plt.show()

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        wandb.log({"pretraining_loss": loss})
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    # def configure_optimizers(self):
    #     optim = torch.optim.Adam(self.parameters(), lr=1e-5)
    #     return optim

    def set_params(self, lr_factor, max_epochs):
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

    def configure_optimizers(self):
        param = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * self.lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [cosine_scheduler]


class Classifier(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.feature_extractor = model
        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

class Supervised_trainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_acc', acc, prog_bar=False)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-5)
        return optimizer


def pretrain():
    print("starting pretraining")
    wandb.init(project='unsup pretraining')

    bs = 256
    num_workers = 16
    input_size = 32
    ratio = input_size / 224
    local_patch_size = int(input_size * ratio)
    ratio2 = 256 / 224
    resize_size = int(input_size * ratio2)

    lr_factor = bs / 256
    max_epochs = 100

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # is this important to have?: target_transform=lambda t: 0 (to ignore object detection)
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)

    # image, _ = dataset[0]
    # print(image.shape)
    # image_np = image.numpy()
    # image_np = np.transpose(image_np, (1, 2, 0))
    # plt.imshow(image_np)
    # plt.axis('off')
    # plt.show()

    dino_transform = DINOTransform(global_crop_size=resize_size, local_crop_size=local_patch_size)
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=dino_transform)

    model = DINO()
    model.set_params(lr_factor, max_epochs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

    wandb.finish()

    # we need to reactivate requires grad to perform supervised backpropagation later
    activate_requires_grad(model.student_backbone)
    return model.student_backbone


def create_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("train size: ", len(train_dataset), "validation size: ", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=12)

    return train_loader, val_loader


def supervised_train(model):
    print("starting sup training")
    wandb.init(project='sup training')

    train_loader, val_loader = create_datasets()
    sup_trainer = Supervised_trainer(model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    wandb_logger = WandbLogger(project='sup training', log_model=True)
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=100, devices=1, accelerator=accelerator, logger=wandb_logger)



    # Train the model
    trainer.fit(sup_trainer, train_loader, val_dataloaders=val_loader)

    wandb.finish()


if __name__ == '__main__':
    pretrained_feature_extractor = pretrain()
    pretrained_model = Classifier(pretrained_feature_extractor, 10)
    supervised_train(pretrained_model)
