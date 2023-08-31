import copy

import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

from lightly.data import LightlyDataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

class Classifier(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.feature_extractor = model
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

def cos_finetune(pretrain_samples, fine_tune_samples):

    prob_decay = np.cos(np.linspace(0, np.pi / 2, pretrain_samples))
    prob_decay /= prob_decay.sum()  # Normalize probabilities

    # Sample values based on the probabilities
    sampled_indices = np.random.choice(pretrain_samples, size=fine_tune_samples, p=prob_decay)
    sampled_values = np.linspace(0, np.pi, pretrain_samples)[sampled_indices]

    # Sample values based on the probabilities
    sampled_indices = np.random.choice(pretrain_samples, size=fine_tune_samples, p=prob_decay)
    sampled_values = np.linspace(0, np.pi, pretrain_samples)[sampled_indices]
    sampled_values = (1 - sampled_values / np.pi) * pretrain_samples
    sampled_values = sampled_values.round()
    occurrences = Counter(sampled_values)
    return occurrences

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
input_dim = 512
# instead of a resnet you can also use a vision transformer backbone as in the
# original paper (you might have to reduce the batch size in this case):
# backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
# input_dim = backbone.embed_dim

model = DINO(backbone, input_dim)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

classifier = Classifier(backbone, 10)
classifier.to(device)

cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
# is this important to have?: target_transform=lambda t: 0 (to ignore object detection)
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
sup_dataset = copy.deepcopy(dataset)

sup_loader = torch.utils.data.DataLoader(
    sup_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8
)


dino_transform = DINOTransform(global_crop_size=224, local_crop_size=224)
dataset = LightlyDataset.from_torch_dataset(dataset, transform=dino_transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = DINOLoss(
    output_dim=2048,
    warmup_teacher_temp_epochs=5,
)
# move loss to correct device because it also contains parameters
criterion = criterion.to(device)

sup_criterion = nn.CrossEntropyLoss()
sup_criterion = sup_criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
sup_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

epochs = 10
sup_epochs = 10

total_steps = len(dataloader) * (epochs + sup_epochs)
sup_counter = cos_finetune(len(dataloader)*epochs, len(sup_loader)*sup_epochs)

sup_loader = [x[1] for x in enumerate(sup_loader)]

sup_steps_done = 0
print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)

    total_sup_loss = 0

    for index, batch in enumerate(dataloader):

        print(f"batch: {index} / {len(dataloader)}")

        print("pretrain step")
        views = batch[0]
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
        update_momentum(model.student_head, model.teacher_head, m=momentum_val)
        views = [view.to(device) for view in views]
        global_views = views[:2]
        teacher_out = [model.forward_teacher(view) for view in global_views]
        student_out = [model.forward(view) for view in views]
        loss = criterion(teacher_out, student_out, epoch=epoch)
        total_loss += loss.detach()
        loss.backward()
        # We only cancel gradients of student head.
        model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
        optimizer.step()
        optimizer.zero_grad()

        if float(index) in sup_counter:
            num_sup_steps = sup_counter[index]
            for sup_step in range(num_sup_steps):
                sup_inputs = sup_loader[sup_steps_done % len(sup_loader)][0].to(device)
                sup_labels = sup_loader[sup_steps_done % len(sup_loader)][1].to(device)
                sup_steps_done += 1
                print("supervised step: ", sup_steps_done)
                sup_optimizer.zero_grad()
                outputs = classifier(sup_inputs)  # Forward pass
                sup_loss = sup_criterion(outputs, sup_labels)  # Compute the loss
                sup_loss.backward()  # Backpropagation
                sup_optimizer.step()  # Update weights

        total_loss += loss.item()



    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")