

import argparse
import torch
import numpy as np
# from model_cnn_shiyan import CNN
# from model_densenet import densenet121
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from utils import read_split_data, train_one_epoch, evaluate
from my_dataset import MyDataSet
# from model_vgg import vgg

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OutputCodeClassifier 
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,recall_score,f1_score,precision_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import confusion_matrix





    







def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument(
        "--model", default='1', type=str, help="[1] CNN-Softmax, [2] CNN-SVM"
    )
    parser.add_argument(

        "--penalty_parameter",
        type=int,
        default=1,
        help="the SVM C penalty parameter",
    )

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/home/wbh/Efficientnet/datasetsM/")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    # parser.add_argument('--weights', type=str, default="/home/wbh/Efficientnet/densenetweightM/densemodel-95.pt",
    #                     help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    arguments = parser.parse_args()
    return arguments

def get_dataset(args):
   
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
    # # 实例化训练数据集
    # dataset_train = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # # 实例化验证数据集
    # dataset_test = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])


    # dataset_train_loader = DataLoader(dataset_train, args.batch_size, shuffle=True)
    # dataset_test_loader = DataLoader(dataset_test, 1, shuffle=False)

    # return dataset_train_loader, dataset_test_loader
    datapath="/home/wbh/Efficientnet/datasetsM/"
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = datasets.ImageFolder(root=os.path.join(datapath, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(datapath, "val"),
                                       
                                            transform=data_transform["val"])
   
    
    dataset_train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dataset_test_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    return dataset_train_loader, dataset_test_loader


def test(model, net1,net2,dataset_test_loader):

    net1.eval()
    net2.eval()
    # clf.decision_function_shape = "ovr"
    with torch.no_grad():
        features = None
        Y = None
        for batch_idx, (images, labels) in enumerate(dataset_test_loader):
            images, labels = images.to(args.device), labels
            feature1 = net1(images)['feature']
            feature2 = net2(images)['feature']
            feature=torch.cat((feature1,feature2),1).cpu().numpy()
            # feature1 = net1(images)['feature'].cpu().numpy()
            # size=feature1.shape[0]

            if features is None:
                features = feature
                Y = labels.numpy()
            else:
                features = np.concatenate((features,feature))
                Y = np.concatenate((Y,labels.numpy()))

            
        acc=model.score(features, Y)
        y_predpro = model.predict_proba(features)
        y_pred=model.predict(features)
        # y_prob=softmax(y_pred,axis=1)
        # auc_score = roc_auc_score(Y, y_prob,multi_class="ovr")
        auc_score = roc_auc_score(Y, y_predpro,multi_class="ovr")
        recall = recall_score(Y, y_pred, average='macro')
        precision= precision_score(Y, y_pred, average='macro')
        f1=f1_score(Y,y_pred,average='macro')

#         # 将标签进行二进制编码
#         y_test_bin = label_binarize(Y, classes=list(range(7)))
#         acc=model.score(features, Y)
#         # y_predpro = model.predict_proba(features)
#         y_pred=model.predict(features)
#         recall = recall_score(Y, y_pred, average='macro')
#         precision= precision_score(Y, y_pred, average='macro')
#         f1=f1_score(Y,y_pred,average='macro')
#         y_test_bin = label_binarize(Y, classes=list(range(7)))

#         auc_score = []
#         for i in range(7):
#             auc_i = roc_auc_score(y_test_bin[:, i], np.where(y_pred == i, 1, 0))
#             auc_score.append(auc_i)

# # 计算平均AUC
#         auc_score = np.mean(auc_score)


    
 
        obj1 = confusion_matrix(Y,y_pred=y_pred)
    
    return acc,auc_score,recall,f1,precision,obj1

def train(net1,net2, args, dataset_train_loader, dataset_test_loader):
# def train(net1, args, dataset_train_loader, dataset_test_loader):


    net1.to(args.device)
    net1.eval()
    net2.to(args.device)
    net2.eval()

    features = None
    Y = None
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataset_train_loader):
            images, labels = images.to(args.device), labels
            net1.zero_grad()
            feature1 = net1(images)['feature']
            feature2 = net2(images)['feature']
            feature=torch.cat((feature1,feature2),1).cpu().numpy()
            # feature1 = net1(images)['feature'].cpu().numpy()
            
            # size=feature1.shape[0]
            
            if features is None:
                features = feature
                Y = labels.numpy()
            else:
                features = np.concatenate((features,feature))
                Y = np.concatenate((Y,labels.numpy()))

            # if  features is None:
            #     #  features =  X_pca.cpu().numpy()
            #     features = feature
            #     Y = labels.numpy()
            # else:
            #     # features = np.concatenate((features,X_pca.cpu().numpy()))
            #     features = np.concatenate((features,feature))
            #     Y = np.concatenate((Y,labels.numpy()))

            # if  features is None:
            #     #  features =  X_pca.cpu().numpy()
            #     features = feature1
            #     Y = labels.numpy()
            # else:
            #     # features = np.concatenate((features,X_pca.cpu().numpy()))
            #     features = np.concatenate((features,feature1))
            #     Y = np.concatenate((Y,labels.numpy()))


    # model = SVC(kernel='rbf', decision_function_shape='ovr',probability=True)#rbf
    
    # model = SVC(kernel='rbf', decision_function_shape='ovo')#rbf
    
    # model = LogisticRegression(random_state=42,max_iter=10000) 
    # ecoc = OutputCodeClassifier(model, code_size=20, random_state=42).fit(features, Y) 

    svm_clf = SVC(kernel='rbf', decision_function_shape='ovo',probability=True)#rbf  
    svm_clf.fit(features, Y)

    # sgd_clf = SGDClassifier(loss = "log_loss")
    # sgd_clf.fit(features, Y)
    
    # Bay_clf=BernoulliNB(alpha=2)
    # Bay_clf.fit(features, Y)

    # GaussianNB_clf=GaussianNB()
    # GaussianNB_clf.fit(features, Y)

    # Multinomial=MultinomialNB(alpha=1).fit(features, Y)

    #决策树
    # dtc = DecisionTreeClassifier(criterion='gini', max_depth=6)
    # dtc.fit(features, Y)

    # 随机森林
    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_classifier.fit(features, Y)

    #K邻
    # KNN = KNeighborsClassifier(n_neighbors = 100, metric="chebyshev")
    # KNN.fit(features,Y)

    # print(test(clf, net1,net2, dataset_test_loader))
    # print(test(ecoc, net1,net2, dataset_test_loader))
    # print(test(sgd_clf, net1,net2, dataset_test_loader))
    # print(test(Bay_clf, net1,net2, dataset_test_loader))
    # print(test(Multinomial, net1,net2, dataset_test_loader))
    # print(test(dtc, net1,net2, dataset_test_loader))
    # print(test(KNN, net1,net2, dataset_test_loader))

    acc,auc_score,recall,f1,precision,obj1=test(svm_clf,net1,net2, dataset_test_loader)
    print('accuracy:{}'.format(acc))
    print('auc:{}'.format(auc_score))
    print('recall:{}'.format(recall))
    print('f1:{}'.format(f1))
    print('precision:{}'.format(precision))
    print(obj1)
    # print(test(ecoc, net1, dataset_test_loader))
    # print(test(clf, net1, dataset_test_loader))




if __name__ == "__main__":
    args = parse_args()
    
    datasets_train_loader, datasets_test_loader = get_dataset(args)

   
    model1 = torch.load("/home/wbh/Efficientnet/densenetweightM/densemodel-95.pt",map_location='cuda:0')
    model2=torch.load("/home/wbh/Efficientnet/mobilenetweightM/mobilemodel-90.pt",map_location='cuda:0')
    #vgg
    # model2 = torch.load("/home/wbh/Efficientnet/vggweight/T/vggmodel-84.pt",map_location='cuda:0')
    start=time.time()
    train(model1,model2, args, datasets_train_loader, datasets_test_loader)
    end=time.time()
    
    print("运行时间"+str(end-start))
    # train(model1, args, datasets_train_loader, datasets_test_loader)



         