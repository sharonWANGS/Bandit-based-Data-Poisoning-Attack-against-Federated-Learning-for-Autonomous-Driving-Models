import matplotlib
matplotlib.use('Agg')
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init
from data import UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView, DAVE2
import torch.optim as optim
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import csv
from os import path
#from scipy.misc import imread, imresize, imsave
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
#from adv_training import test_on_file
import copy
import random

device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cent_train(model_name='baseline', epochs=50, train = 1):
    batch_size = 32
    lr = 0.0001
    dataset_path = './data/Udacity/'
    setup_seed(1)
    device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
    print(device)

    if model_name == 'baseline':
        net = BaseCNN()
    elif model_name == 'nvidia':
        net = Nvidia()
    elif model_name == 'vgg16':
        net = Vgg16()
    print(model_name)
    resized_image_height = 128
    resized_image_width = 128
    net.apply(weight_init)
    net = net.to(device)
    # net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

    image_size = (resized_image_width, resized_image_height)

    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))

        composed = transforms.Compose(
            [Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
        dataset = UdacityDataset(dataset_path+'training/', ['HMB1','HMB2', 'HMB4', 'HMB5', 'HMB6'], composed)
        # dataset = DAVE2(dataset_path, composed)
        steps_per_epoch = int(len(dataset) / batch_size)

        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        if model_name == 'vgg16':
            optimizer = optim.Adam(net.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr=lr)

        # x,y = train_generator.__next__()
        # print(x.shape)
        loss_list = []
        rmse_round = []
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            for step, sample_batched in enumerate(train_generator):
                if step <= steps_per_epoch:
                    batch_x = sample_batched['image']
                    # print(batch_x.numpy())
                    batch_y = sample_batched['steer']

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = net(batch_x)

                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                else:
                    break
            print('Epoch %d  training loss: %.4f' % (epoch, total_loss / steps_per_epoch))
            loss_list.append(total_loss / steps_per_epoch)
            end = time.time()
            print(end-start)

            net.eval()
            with torch.no_grad():
                yhat = []
                # test_y = []
                test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values

                test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
                test_dataset = UdacityDataset('./data/Udacity', ['testing'], test_composed, 'test')
                # test_dataset = DAVE2(dataset_path, test_composed, 'test')

                test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
                for _, sample_batched in enumerate(test_generator):
                    batch_x = sample_batched['image']
                    # print(batch_x.size())
                    # print(batch_x.size())
                    batch_y = sample_batched['steer']
                    # print(batch_y)

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    output = net(batch_x)

                    yhat.append(output.item())

                yhat = np.array(yhat)
                rmse = np.sqrt(np.mean((yhat - test_y) ** 2))
                print(rmse)
                rmse_round.append(rmse)
            end1 = time.time()
            print(end1-start)


        test_results = pd.DataFrame(data=rmse_round)
        test_results.to_csv('./results/centrtrain_model/' + model_name + '_rmse.csv')
        plt.plot(loss_list)
        plt.xlabel('training round', fontsize=20)
        plt.ylabel('test RMSE', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.savefig('./results/centrtrain_model/' + model_name + '_rmse.png')

        # torch.save(net.state_dict(), './checkpoints/' + model_name + '.pt')
        # loss_dataframe = pd.DataFrame(data=loss_list)
        # loss_dataframe.to_csv('./results/centrtrain_model/train_loss.csv')
        # plt.plot(loss_list)
        # plt.yticks(size=14)
        # plt.xticks(size=14)
        # plt.title("training loss", fontsize=20)
        # plt.savefig('./results/centrtrain_model/' + model_name + '_train_loss.png')
    else:
        net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

    net.eval()
    with torch.no_grad():
        yhat = []
        # test_y = []
        test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values

        test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
        test_dataset = UdacityDataset('./data/Udacity', ['testing'], test_composed, 'test')
        # test_dataset = DAVE2(dataset_path, test_composed, 'test')

        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for _, sample_batched in enumerate(test_generator):
            batch_x = sample_batched['image']
            # print(batch_x.size())
            # print(batch_x.size())
            batch_y = sample_batched['steer']
            # print(batch_y)

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)

            yhat.append(output.item())

        yhat = np.array(yhat)
        yhat_dataframe = pd.DataFrame(data=yhat)
        test_y_dataframe = pd.DataFrame(data=test_y)
        yhat_dataframe.to_csv('./results/centrtrain_model/'+model_name+'_test_rmse.csv')
        # print(yhat)
        rmse = np.sqrt(np.mean((yhat - test_y) ** 2))
        print(rmse)
        plt.figure(figsize=(32, 8))
        plt.plot(test_y, 'r.-', label='Label_steer')
        plt.plot(yhat, 'b^-', label='Output_steer')
        plt.legend(prop={'size': 16})
        plt.xlabel('test images', fontsize=20)
        plt.ylabel('steer angle', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.title("RMSE: %.2f" % rmse, fontsize=20)
        # plt.show()
        model_fullname = "%s_test_rmse.png" % (model_name)
        plt.savefig('./results/centrtrain_model/' + model_fullname)

def pretrain(model_name='nvidia', epochs=50, train = 1):
    batch_size = 32
    lr = 0.0001
    dataset_path = './data/Udacity/'
    setup_seed(1)
    device = torch.device('cuda:6' if torch.cuda.is_available() else "cpu")
    print(device)

    if model_name == 'baseline':
        net = BaseCNN()
    elif model_name == 'nvidia':
        net = Nvidia()
    elif model_name == 'vgg16':
        net = Vgg16()
    print(model_name)
    resized_image_height = 128
    resized_image_width = 128
    net.apply(weight_init)
    net = net.to(device)
    # net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

    image_size = (resized_image_width, resized_image_height)

    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))

        composed = transforms.Compose(
            [Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
        dataset = UdacityDataset(dataset_path+'training/', ['HMB1'], composed)
        # dataset = DAVE2(dataset_path, composed)
        steps_per_epoch = int(len(dataset) / batch_size)

        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        if model_name == 'vgg16':
            optimizer = optim.Adam(net.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr=lr)

        # x,y = train_generator.__next__()
        # print(x.shape)
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0
            for step, sample_batched in enumerate(train_generator):
                if step <= steps_per_epoch:
                    batch_x = sample_batched['image']
                    # print(batch_x.numpy())
                    batch_y = sample_batched['steer']

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = net(batch_x)

                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                else:
                    break
            print('Epoch %d  RMSE loss: %.4f' % (epoch, total_loss / steps_per_epoch))
            loss_list.append(total_loss / steps_per_epoch)

        torch.save(net.state_dict(), './checkpoints/' + model_name + '.pt')
        loss_dataframe = pd.DataFrame(data=loss_list)
        loss_dataframe.to_csv('./results/pretrained_model/train_loss.csv')
        plt.plot(loss_list)
        plt.legend(prop={'size': 16})
        plt.xlabel('training round', fontsize=20)
        plt.ylabel('training loss', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        # plt.title("training loss", fontsize=20)
        plt.savefig('./results/pretrained_model/' + model_name + '_train_loss.png')
    else:
        net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

    net.eval()
    with torch.no_grad():
        yhat = []
        # test_y = []
        test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values

        test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
        test_dataset = UdacityDataset('./data/Udacity', ['testing'], test_composed, 'test')
        # test_dataset = DAVE2(dataset_path, test_composed, 'test')

        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for _, sample_batched in enumerate(test_generator):
            batch_x = sample_batched['image']
            # print(batch_x.size())
            # print(batch_x.size())
            batch_y = sample_batched['steer']
            # print(batch_y)

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)

            yhat.append(output.item())

        yhat = np.array(yhat)
        yhat_dataframe = pd.DataFrame(data=yhat)
        test_y_dataframe = pd.DataFrame(data=test_y)
        yhat_dataframe.to_csv('./results/pretrained_model/'+model_name+'_test_rmse.csv')
        # print(yhat)
        rmse = np.sqrt(np.mean((yhat - test_y) ** 2))
        print(rmse)
        plt.figure(figsize=(32, 8))
        plt.plot(test_y, 'r.-', label='Label_steer')
        plt.plot(yhat, 'b^-', label='Output_steer')
        plt.legend(prop={'size': 16})
        plt.xlabel('test images', fontsize=20)
        plt.ylabel('steer angle', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.title("RMSE: %.2f" % rmse, fontsize=20)
        # plt.show()
        model_fullname = "%s_test_rmse.png" % (model_name)
        plt.savefig('./results/pretrained_model/' + model_fullname)

def cent_train_splitdata(model_name='baseline', epochs=50, train = 1):
    batch_size = 32
    lr = 0.0001
    dataset_path = './data/Udacity/'
    setup_seed(1)
    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
    print(device)

    if model_name == 'baseline':
        net = BaseCNN()
    elif model_name == 'nvidia':
        net = Nvidia()
    elif model_name == 'vgg16':
        net = Vgg16()
    print(model_name)
    resized_image_height = 128
    resized_image_width = 128
    net.apply(weight_init)
    net = net.to(device)
    # net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

    image_size = (resized_image_width, resized_image_height)

    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))

        composed = transforms.Compose(
            [Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
        train_dataset = UdacityDataset(dataset_path + 'training/', ['HMB1','HMB2', 'HMB4', 'HMB5', 'HMB6'], transform=composed)

        datasize = []
        for group in range(epochs):
            if group < (epochs) - 1:
                datasize.append(int(train_dataset.data.shape[0] / epochs))
            else:
                datasize.append(train_dataset.data.shape[0] - sum(datasize))
        print(train_dataset.data.shape[0], datasize)

        traindata_split = torch.utils.data.random_split(train_dataset, datasize)
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False) for x in traindata_split]
        steps_per_epoch = int(len(train_dataset) / batch_size)

        # dataset = UdacityDataset(dataset_path+'training/', ['HMB1','HMB2', 'HMB4', 'HMB5', 'HMB6'], composed)
        # dataset = DAVE2(dataset_path, composed)

        # train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        if model_name == 'vgg16':
            optimizer = optim.Adam(net.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr=lr)

        # x,y = train_generator.__next__()
        # print(x.shape)
        loss_list = []
        rmse_round = []
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0

            for step, sample_batched in enumerate(train_loader[epoch]):
                if step <= steps_per_epoch:
                    batch_x = sample_batched['image']
                    # print(batch_x.numpy())
                    batch_y = sample_batched['steer']

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = net(batch_x)

                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                else:
                    break
            print('Epoch %d  training loss: %.4f' % (epoch, total_loss / steps_per_epoch))
            loss_list.append(total_loss / steps_per_epoch)
            end = time.time()
            print(end-start)

            net.eval()
            with torch.no_grad():
                yhat = []
                # test_y = []
                test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values

                test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
                test_dataset = UdacityDataset('./data/Udacity', ['testing'], test_composed, 'test')
                # test_dataset = DAVE2(dataset_path, test_composed, 'test')

                test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
                for _, sample_batched in enumerate(test_generator):
                    batch_x = sample_batched['image']
                    # print(batch_x.size())
                    # print(batch_x.size())
                    batch_y = sample_batched['steer']
                    # print(batch_y)

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    output = net(batch_x)

                    yhat.append(output.item())

                yhat = np.array(yhat)
                rmse = np.sqrt(np.mean((yhat - test_y) ** 2))
                print(rmse)
                rmse_round.append(rmse)
            end1 = time.time()
            print(end1-start)


        test_results = pd.DataFrame(data=rmse_round)
        test_results.to_csv('./results/centrtrain_model/' + model_name + '_splitdata_rmse.csv')
        plt.plot(loss_list)
        plt.xlabel('training round', fontsize=20)
        plt.ylabel('test RMSE', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.savefig('./results/centrtrain_model/' + model_name + '_splitdata_rmse.png')

        # torch.save(net.state_dict(), './checkpoints/' + model_name + '.pt')
        # loss_dataframe = pd.DataFrame(data=loss_list)
        # loss_dataframe.to_csv('./results/centrtrain_model/train_loss.csv')
        # plt.plot(loss_list)
        # plt.yticks(size=14)
        # plt.xticks(size=14)
        # plt.title("training loss", fontsize=20)
        # plt.savefig('./results/centrtrain_model/' + model_name + '_train_loss.png')
    else:
        net.load_state_dict(torch.load('./checkpoints/' + model_name + '_splitdata.pt'))

    net.eval()
    with torch.no_grad():
        yhat = []
        # test_y = []
        test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values

        test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
        test_dataset = UdacityDataset('./data/Udacity', ['testing'], test_composed, 'test')
        # test_dataset = DAVE2(dataset_path, test_composed, 'test')

        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for _, sample_batched in enumerate(test_generator):
            batch_x = sample_batched['image']
            # print(batch_x.size())
            # print(batch_x.size())
            batch_y = sample_batched['steer']
            # print(batch_y)

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)

            yhat.append(output.item())

        yhat = np.array(yhat)
        yhat_dataframe = pd.DataFrame(data=yhat)
        test_y_dataframe = pd.DataFrame(data=test_y)
        yhat_dataframe.to_csv('./results/centrtrain_model/'+model_name+'_splitdata_test_rmse.csv')
        # print(yhat)
        rmse = np.sqrt(np.mean((yhat - test_y) ** 2))
        print(rmse)
        plt.figure(figsize=(32, 8))
        plt.plot(test_y, 'r.-', label='Label_steer')
        plt.plot(yhat, 'b^-', label='Output_steer')
        plt.legend(prop={'size': 16})
        plt.xlabel('test images', fontsize=20)
        plt.ylabel('steer angle', fontsize=20)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.title("RMSE: %.2f" % rmse, fontsize=20)
        # plt.show()
        model_fullname = "%s_splitdata_test_rmse.png" % (model_name)
        plt.savefig('./results/centrtrain_model/' + model_fullname)

if __name__ == "__main__":
    global_model = cent_train_splitdata(model_name='baseline', epochs=300, train = 1)
    # global_model = pretrain(model_name='nvidia', epochs=50, train = 1)