import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


import numpy as np
# from model_densenet import  densenet121,load_state_dict
from mobilenetv3 import mobilenet_v3_large

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
import json




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # model_choice = args.model
    # assert (
    #     model_choice == "1" or model_choice == "2"
    # ), "Invalid choice: Choose between 1 and 2 only."

    # if model_choice == "1":         # 44%
    #     model = densenet121(num_classes=args.num_classes)
    # elif model_choice == "2":       # 41%
    #     model = CNN_SVM()
    # print(model)

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    image_path=args.data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    train_num = len(train_dataset)

    # {'O':0, 'NE':1, 'EV':2,'EEC':3,'AEC':4}
    disease_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in disease_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    #json_str = json.dumps(cla_dict, indent=2)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 4])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
                                            #    collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
                                            #  collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    # model = densenet121(num_classes=args.num_classes).to(device)
    model = mobilenet_v3_large(num_classes=args.num_classes).to(device)
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         load_state_dict(model, args.weights)
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))

    assert os.path.exists(args.weights), "file {} dose not exist.".format(args.weights)
    pre_weights = torch.load(args.weights, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)


    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model, "/data/wyx/densenetweight2Tdata/densemodel-{}.pt".format(epoch))
        torch.save(model, "/data/wyx/mobiletweight2Tdata/mobilemodel-{}.pt".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/wyx/datasetsTpick/")
    parser.add_argument(
        "--model", default='1', type=str, help="[1] CNN-Softmax, [2] CNN-SVM"
    )
    parser.add_argument(

        "--penalty_parameter",
        type=int,
        default=1,
        help="the SVM C penalty parameter",
    )

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    # parser.add_argument('--weights', type=str, default="/home/wyx/efficientnet/densenet/densenet121-a639ec97.pth",
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default="/home/wyx/mobilenetv3/mobilenet_v3_large-8738ca79.pth",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
