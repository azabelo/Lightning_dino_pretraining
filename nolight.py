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


cifar_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
# is this important to have?: target_transform=lambda t: 0 (to ignore object detection)
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)

    index = 0
    for batch in dataloader:
        print(f"batch: {index} / {len(dataloader)}")
        index += 1

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

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")