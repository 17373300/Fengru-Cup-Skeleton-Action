import numpy as np
import pickle
import os
from NTU_RGB_D_Dataset import NTU_RGB_D_Dataset
from torch.utils.data import DataLoader
from Models.st_gcn import ST_GCN_18
import torch
import random
import matplotlib.pyplot as plt

train_mode = False  # True: train mode; False: test mode using pretrained model
num_epoch = 80
batch_size = 2
test_batch_size = 64
early_stop_alpha = 5
num_class = 60

dataset_types = ['xview', 'xsub']


def train(model, train_data, train_label):
    assert isinstance(model, ST_GCN_18)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.0001)

    # divide dataset into training and validation
    dataset = NTU_RGB_D_Dataset(data=train_data, label=train_label)
    val_dataset_size = int(0.1 * len(dataset))
    val_dataset, train_dataset = torch.utils.data.random_split(dataset,
                                                               [val_dataset_size, len(dataset) - val_dataset_size])

    train_dataLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
    val_dataLoader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=4)

    loss_func = torch.nn.CrossEntropyLoss()
    model.train()

    min_val_loss = None
    print('******** start to train ********')
    for epoch in range(num_epoch):
        for step, (batch_x, batch_y) in enumerate(train_dataLoader):
            pred_y = model(batch_x)
            loss = loss_func(pred_y, batch_y)
            loss.backward()
            optimizer.zero_grad()

        # validate
        with torch.no_grad():
            right_cases, val_loss = 0, 0
            for batch_x, batch_y in val_dataLoader:
                pred_y = model(batch_x)
                right_cases += (torch.argmax(pred_y, dim=1) == batch_y.unsqueeze(1)).sum()
                val_loss += loss_func(pred_y, batch_y)
            accuracy = right_cases / len(val_dataset)
            print(
                '\033[1;35m val:\033[0m epoch: %d | loss: %.3f | accuracy: %.3f%%' % (epoch, val_loss, accuracy * 100))

            if min_val_loss is None:
                min_val_loss = val_loss
            else:
                if val_loss > min_val_loss:
                    GL = 100 * (val_loss / min_val_loss - 1)
                    if GL > early_stop_alpha:
                        # early stop
                        print('\033[1;35m early stop\033[0m')
                        break
                else:
                    min_val_loss = val_loss

        lr_scheduler.step(epoch)
    return model


def test(model, test_data, test_label):
    model.eval()
    test_dataset = NTU_RGB_D_Dataset(data=test_data, label=test_label)
    test_dataLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False,
                                                  num_workers=4)
    loss_func = torch.nn.CrossEntropyLoss()
    print('******** start to test ********')
    with torch.no_grad():
        right_cases, test_loss = 0, 0
        for step, (batch_x, batch_y) in enumerate(test_dataLoader):
            pred_y = model(batch_x)
            # print('size of pred_y: %d | size of batch_y: %d' % (len(pred_y), len(batch_y)))
            right_cases += (torch.argmax(pred_y, dim=1) == batch_y.unsqueeze(1)).sum()
            test_loss += loss_func(pred_y, batch_y)
            cur_accuracy = right_cases / ((step + 1) * test_batch_size)
            print('\033[1;35m test:\033[0m step %d finishes | test loss: %.3f | current accuracy: %.2f%%' % (
                step, test_loss, cur_accuracy * 100))
        accuracy = right_cases / len(test_dataset)
        print('\033[1;35m test:\033[0m loss: %.3f | accuracy: %.3f%%' % (test_loss, accuracy * 100))


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    random.seed(11)
    for type in dataset_types:
        print('******* %s start *******' % type)
        folderName = os.path.join('data', 'NTU-RGB+D', type)
        train_data_fileName = os.path.join(folderName, 'train_data.npy')
        train_label_fileName = os.path.join(folderName, 'train_label.pkl')
        test_data_fileName = os.path.join(folderName, 'val_data.npy')
        test_label_fileName = os.path.join(folderName, 'val_label.pkl')

        train_data = np.load(train_data_fileName)
        with open(train_label_fileName, 'rb') as f1:
            train_label = pickle.load(f1)

        test_data = np.load(test_data_fileName)
        with open(test_label_fileName, 'rb') as f2:
            test_label = pickle.load(f2)

        model = ST_GCN_18(in_channels=3,
                          num_class=num_class,
                          edge_importance_weighting=True,
                          graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                          dropout=0.5
                          )
        if torch.cuda.is_available():
            model = model.cuda()

        if train_mode:
            model = train(model, train_data, train_label)
            torch.save(model.state_dict(), os.path.join('checkpoints', 'my_st_gcn.ntu-%s.pth' % type))
        else:
            # load pretrained model
            pth_file = os.path.join('checkpoints', 'st_gcn.ntu-%s.pth' % type)
            model.load_state_dict(torch.load(pth_file))

        # test
        test(model, test_data, test_label)
