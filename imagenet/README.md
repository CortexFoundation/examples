# ImageNet training in PyTorch(Modified By DeepInsight)

The original trunk implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

We implement WideResNet and ResNeXt.

We add multiple hyperparameters like channel-size, some second-order tricks, and which is we think most important -- modifiable expansion rate.


We achieve 84.36% Top-1 Acc on CIFAR100 with single crop in validation set.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders

## Training

To train a conventional model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

To train a wider resnext with multiple expansion rate, try:

```bash
python main_next.py --arch resnext29_cifar100 --ds CIFAR100 --batch-size 128 --x 80 --d 32 --xp 0.25 --wd 0.001 --nes 0 --df 0 --lr 0.05 --lp 150 --epochs 400 ./
```


## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | resnet | resnet101 |
                        resnet152 | resnet18 | resnet34 | resnet50 | vgg |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --x                   channel nums
  --d                   channel size
  --xp                  expansion rate
  --df                  Use Deformable ConvNets
  --lp                  The first time learning rate start to decrease
```
