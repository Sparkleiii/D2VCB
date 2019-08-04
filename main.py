import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import myDataSet as ms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import load_data as ld
from models import BCL_Network
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, \
    precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold
import util


# from tensorboardX import SummaryWriter


# 返回每一折多个epoch中的最优模型
def train(myDataLoader, path, fold):
    best = 0
    for epoch in range(Epoch):
        for step, (x, y) in enumerate(myDataLoader):
            model.train()
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ROC, PR, F1, test_loss, accuracy = validate(validate_DataLoader, epoch)
        if ROC > best:
            best = ROC
            model_name = ld.modelDir + path + '/validate_params_' + str(fold) + '.pkl'
            torch.save(model.state_dict(), model_name)
    scheduler.step(test_loss)
    print(model_name)
    return model_name


def validate(myDataLoader, epoch):
    output_list = []
    output_result_list = []
    correct_list = []
    test_loss = 0
    for step, (x, y) in enumerate(myDataLoader):
        model.eval()
        output = model(x)
        loss = loss_func(output, y)
        test_loss += float(loss)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    y_pred = np.array(output_result_list)
    y_true = np.array(correct_list)
    accuracy = accuracy_score(y_true, y_pred)
    test_loss /= myDataLoader.__len__()
    print('Validate set: Average loss:{:.4f}\tAccuracy:{:.3f}'.format(test_loss, accuracy))
    ROC, PR, F1 = util.get_ROC_Curve(output_list, output_result_list, correct_list)
    print('第{}折_第{}轮_ROC:{}\tPR:{}\tF1:{} '.format(fold, epoch, ROC, PR, F1))
    return ROC, PR, F1, test_loss, accuracy


def test(myDataLoader, path, fold, best_model_name):
    name = 'validate_params_' + str(fold)
    model.load_state_dict(torch.load(best_model_name))
    output_list = []
    output_result_list = []
    correct_list = []
    for step, (x, y) in enumerate(myDataLoader):
        model.eval()
        output = model(x)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list, path + '/' + name)
    return ROC, PR, F1


def getDataSet(train_index, test_index):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]
    x_train_, x_validate_, y_train_, y_validate_ = train_test_split(
        x_train, y_train, test_size=0.125, stratify=y_train)
    x_train_ = x_train_.reset_index(drop=True)
    x_validate_ = x_validate_.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train_ = y_train_.reset_index(drop=True)
    y_validate_ = y_validate_.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_DataSet = ms.MyDataSet(input=x_train_, label=y_train_)
    validate_DataSet = ms.MyDataSet(input=x_validate_, label=y_validate_)
    test_DataSet = ms.MyDataSet(input=x_test, label=y_test)
    train_DataLoader = DataLoader(dataset=train_DataSet, batch_size=Batch_Size, shuffle=True)
    validate_DataLoader = DataLoader(dataset=validate_DataSet, batch_size=test_Batch_Size, shuffle=True)
    test_DataLoader = DataLoader(dataset=test_DataSet, batch_size=test_Batch_Size, shuffle=True)
    return train_DataLoader, validate_DataLoader, test_DataLoader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    now_time = datetime.date.today()
    # result_file = open(str(now_time) + '_list1.txt', 'a')
    result_file = open(str(now_time) + '.txt', 'a')
    # 常用参数
    Batch_Size = 64
    test_Batch_Size = 128
    LR = 0.001
    Epoch = 15
    K_Fold = 3
    print("Batch_Size", Batch_Size)
    print("Epoch", Epoch)
    print("K_Fold", K_Fold)
    file_list = ld.create_list(ld.dataDir)
    file_list.sort()
    # file_list = ['wgEncodeHaibTfbsGm12878Egr1V0416101PkRep1']
    file_list = ['wgEncodeHaibTfbsGm12878Bcl3V0416101PkRep1', 'wgEncodeHaibTfbsGm12878Mef2aPcr1xPkRep1']
    for path in file_list:
        all_data = pd.read_csv(ld.dataDir + path + '/all_data.txt', sep='\t')
        X = all_data['sequence']
        y = all_data['label']
        kf = StratifiedKFold(n_splits=K_Fold, shuffle=True)
        fold = 1
        roc_total = []
        pr_total = []
        F1_total = []
        for train_index, validate_index in kf.split(X, y):
            # writer = SummaryWriter(comment='test')
            train_DataLoader, validate_DataLoader, test_DataLoader = getDataSet(train_index, validate_index)
            model = BCL_Network().to(device)
            # model = nn.parallel.DataParallel(model, device_ids=[0, 1, 2, 3])
            #  优化器和损失函数writer
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            # 动态学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
            loss_func = nn.BCELoss().to(device)
            best_model_name = train(train_DataLoader, path, fold)
            # best_model_name = '/ifs/gdata2/wuhui/ProteinDNABinding/model/wgEncodeHaibTfbsGm12878Egr1V0416101PkRep1
            ROC, PR, F1 = test(test_DataLoader, path, fold, best_model_name)
            roc_total.append(ROC)
            pr_total.append(PR)
            F1_total.append(F1)
            fold += 1
        # 获得三折的平均AUC值
        roc_average = np.mean(roc_total)
        pr_average = np.mean(pr_total)
        f1_average = np.mean(F1_total)
        print(path)
        print("Average ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
        print("#################################")
        result_file.write(path + '\n')
        result_file.write("Average ROC:{}\tPR:{}\tF1:{}\n".format(roc_average, pr_average, f1_average))
    result_file.close()
