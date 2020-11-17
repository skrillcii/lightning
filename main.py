import os
from model import LitAutoEncoder
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

dataset = MNIST(os.getcwd() + '/datasets', download=True,
                transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=60000, num_workers=4)

# init model
autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults
# (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer(gpus=[0, 1], num_nodes=1, accelerator='dp',
                     max_epochs=200, log_every_n_steps=1)
trainer.fit(autoencoder, train_loader)

# train on TPUs using 16 bit precision
# using only half the training data and checking
# validation every quarter of a training epoch
#
# trainer = pl.Trainer(
#         tpu_cores=8,
#         precision=16,
#         limit_train_batches=0.5,
#         val_check_interval=0.25
# )
