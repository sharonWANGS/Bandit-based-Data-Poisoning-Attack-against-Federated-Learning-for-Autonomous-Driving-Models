from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from data import UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView, DAVE2
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init
from bandit import Epsilon_Greedy_Bandit, UCB_Bandit
from attack_update import AttackDataset, client_update, test, server_test, attack_client_update, server_weight_aggregate
from torch.utils.data import Subset
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def PoisonAttack(data_loader, target, min, max, batch_size=32):
    images = []

    targets = []

    steers = torch.zeros(0)
    st_num = 0
    # print('min:',min)
    for _, sample_batched in enumerate(data_loader):
        # Send the data and label to the device
        data = sample_batched['image'].type(torch.FloatTensor).to(device)
        steer = sample_batched['steer'].type(torch.FloatTensor).to(device)

        for st in steer:
            # print(type(st))
            if min <= st and st <= max:
                # print(st)
                st_num += 1
                target_steer = st + target
                # target_steer = torch.FloatTensor([1.0])
            else:
                target_steer = st
            target_steer = target_steer.to(device)
            steers = steers.to(device)
            steers = torch.cat((steers, target_steer), 0)

            # print(steers, type(steers))

        target_steer = steers.to(device)

        # targets = np.concatenate((targets, target_steer.squeeze().detach().cpu().numpy()), axis=0)
        targets = target_steer.squeeze().detach().cpu().numpy()

        if images == []:
            images = data.squeeze().detach().cpu().numpy()
        else:
            images = np.concatenate((images, data.squeeze().detach().cpu().numpy()), axis=0)

    attack_dataset = AttackDataset(images, targets)

    attack_dataloader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return attack_dataloader



def UCB_attack(model_name='baseline', num_subdataset=1, num_clients=1, num_rounds=10, epochs=10,
                                       random_clients=False):
    setup_seed(2)

    target = -0.4
    batch_size = 32
    dataset_path = './data/Udacity/'

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


    global_model = copy.deepcopy(net)
    client_models = [copy.deepcopy(net) for _ in range(num_clients)]
    for model in client_models:  # update client model
        model.load_state_dict(global_model.state_dict())

    image_size = (resized_image_width, resized_image_height)

    composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
    train_dataset = UdacityDataset(dataset_path+'training/', ['HMB1','HMB2', 'HMB4', 'HMB5', 'HMB6'], transform=composed)

    #     traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) if group<num_clients-1 else int(train_dataset.data.shape[0]-train_dataset.data.shape[0] / num_clients)*(num_clients-1)
    #  for group in range(num_clients)])
    datasize = []
    for group in range(num_subdataset):
        if group < (num_subdataset) - 1:
            datasize.append(int(train_dataset.data.shape[0] / num_subdataset))
        else:
            datasize.append(train_dataset.data.shape[0] - sum(datasize))
    print(train_dataset.data.shape[0], datasize)
    # traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) for _ in range(num_clients)])


    traindata_split = torch.utils.data.random_split(train_dataset, datasize)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    num_test = len(test_dataset)

    attack_test_loader = [torch.utils.data.DataLoader(x, batch_size=1, shuffle=False) for x in traindata_split]

    steps_per_epoch = int(int(train_dataset.data.shape[0] / num_subdataset) / batch_size)

    criterion = nn.L1Loss()
    lr = 0.0001
    opt = [optim.Adam(model.parameters(), lr=lr) for model in client_models]
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    client_optimizers = [optim.Adam(client_models[i].parameters(), lr=lr) for i in range(num_clients)]

    rmse_round = []
    attackrate_round = []
    steer_round = []
    output_round = []

    drop = []
    train_ro_cl_ep_losses = []  # train_round_client_epoch_losses

    client_list = [[random.randint(0, num_subdataset - 1) for j in range(num_clients)] for i in range(num_rounds)]
    # print(client_list)

    grad_global_model = global_model
    weight_global_model = global_model

    # drop_index = server_grad_aggregate_detectdistr(grad_global_model, weight_global_model, client_models, optimizer, test_loader, criterion, 0, num_rounds, drop=True)

    # bandit
    step_reward_0 = [0]
    avgacc_reward_0 = [0]
    action_round_0 = [0]
    att_num_round_0 = [0]

    step_reward_1 = [0]
    avgacc_reward_1 = [0]
    action_round_1 = [0]
    att_num_round_1 = [0]
    server_rmse_round = []
    server_test_losses = []
    R_min = 0
    R_max = 0.2
    att_round = 30
    action = 0
    bandit = UCB_Bandit()

    dirs = './results/Paper/UCB/seed2/'
    print(dirs)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for r in range(num_rounds):
        print('Number of rounds: %d' % r)
        start_time = time.time()
        client_idx = client_list[r]

        # client update
        train_client_losses = []
        # attack
        if r < att_round:
            for i in range(num_clients):
                if i == 0:
                    # test_loss, rmse, steers, outputs, R, att_num = test(client_models[i],
                    #                                                     attack_test_loader[client_idx[i]],
                    #                                                     criterion, min=R_min, max=R_max)
                    action_0 = None
                    action_round_0.append(action_0)
                    step_reward_0.append(0)
                    avgacc_reward_0.append(0)
                    att_num_round_0.append(0)

                    train_epoch_losses, train_iter_losses = client_update(client_models[i], client_optimizers[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)
                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], client_optimizers[i], criterion, \
                                                                      train_loader[client_idx[i]], epochs=epochs, \
                                                                      steps_per_epoch=steps_per_epoch)

                train_client_losses.append(train_epoch_losses)

                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))
        else:
            for i in range(num_clients):
                if i == 0:
                    print('attack')
                    if r <= att_round+num_clients-1: # UCB initialization 4rounds
                        if r == att_round:
                            R_min = -2
                            R_max = 2
                            action_round_0.append(None)
                            step_reward_0.append(0)
                            avgacc_reward_0.append(0)
                            att_num_round_0.append(0)
                            attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target,
                                                             min=R_min, max=R_max, batch_size=batch_size)


                            train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], client_optimizers[i],
                                                                                         criterion, \
                                                                                         attack_dataloader,
                                                                                         epochs=epochs)

                            action = 0
                            R_min = 0
                            R_max = 0.2
                        else:
                            action_round_0.append(action)
                            print("action:", action)
                            print('test_loader', client_idx[i])
                            test_loss, rmse, steers, outputs, R, att_num = test(client_models[i],
                                                                        attack_test_loader[client_idx[i]],
                                                                        criterion, min=R_min, max=R_max)
                            bandit.update_est(action, R)
                            step_reward_0.append(R)
                            avgacc_reward_0.append((r * avgacc_reward_0[-1] + R) / (r + 1))
                            att_num_round_0.append(att_num)
                            #print("att_num", att_num_round)

                            if r == att_round + 1:
                                action = 1
                                R_min = 0.2
                                R_max = 0.4
                            elif r == att_round+2:
                                action = 2
                                R_min = 0.4
                                R_max = 0.6
                            elif r == att_round+3:
                                action = 3
                                R_min = 0.6
                                R_max = 1.0
                            # bandit reward
                            attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target,
                                                             min=R_min, max=R_max, batch_size=batch_size)

                            train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], client_optimizers[i], criterion, \
                                                                                         attack_dataloader, epochs=epochs)

                            rmse_round.append(rmse)
                            attackrate_round.append(R)


                            print('--- test loss %0.4f | test rmse: %0.4f | delta: %0.4f' % (test_loss, rmse, R))

                    else:                         #UCB attack after the first 4 rounds
                        action_round_0.append(action)

                        test_loss, rmse, steers, outputs, R, att_num = test(client_models[i], attack_test_loader[client_idx[i]],
                                                                    criterion, min=R_min, max=R_max)
                        att_num_round_0.append(att_num)
                        bandit.update_est(action, R)
                        action = bandit.choose_action(r)
                        print("action:", action)
                        #print("att_num", att_num_round)
                        if action == 0:
                            R_min = 0
                            R_max = 0.2
                        elif action == 1:
                            R_min = 0.2
                            R_max = 0.4
                        elif action == 2:
                            R_min = 0.4
                            R_max = 0.6
                        else:
                            R_min = 0.6
                            R_max = 1.0
                        attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target,
                                                         min=R_min, max=R_max, batch_size=batch_size)

                        # client_models_0_dict = client_models[0].state_dict()
                        # print('after_poisonattack'+str(i), client_models_0_dict['layer4.0.weight'])

                        train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], client_optimizers[i], criterion, \
                                                                                     attack_dataloader, epochs=epochs)


                        step_reward_0.append(R)
                        avgacc_reward_0.append((r * avgacc_reward_0[-1] + R) / (r + 1))
                        # step_reward = step_reward.cpu()
                        # avgacc_reward = avgacc_reward.cpu()
                        print("step reward:", R)
                        #print("avgacc_reward:", avgacc_reward)
                        rmse_round.append(rmse)
                        attackrate_round.append(R)

                        print('--- test loss %0.4f | test rmse: %0.4f | delta: %0.4f' % (test_loss, rmse, R))

                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], client_optimizers[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)


                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))

        train_ro_cl_ep_losses.append(train_client_losses)
        train_loss = np.sum(np.asarray(train_client_losses)) / np.asarray(train_client_losses).size
        # print('--- training loss: %0.4f ---'% (train_loss))
        # print("--- client update time: %0.1f seconds ---" % (time.time() - start_time))

        # server aggregate

        server_weight_aggregate(grad_global_model, client_models, optimizer, drop=False, defense='trimmed_mean')
        #server_grad_aggregate(grad_global_model, weight_global_model, client_models, optimizer, drop=False)

        #test on the server
        # if r%10 == 0:
        server_test_loss, server_rmse, steers, outputs, ast, num_sample = test(global_model, test_loader, criterion, min=-1.0, max=1.0)
        print("server_test_loss:", server_test_loss)
        print("server_rmse:", server_rmse)
        server_test_losses.append(server_test_loss)
        server_rmse_round.append(server_rmse)
        steer_round.append(steers)
        output_round.append(outputs)

        print("--- one iteration time: %0.1f seconds ---" % (time.time() - start_time))



    plt.figure(1)
    plt.figure(3, (32,8))
    plt.plot(step_reward_0, color='red', label='step_reward')
    plt.plot(avgacc_reward_0, color='blue', label='avgacc_reward')
    plt.title('attack reward', fontsize=20)
    plt.legend(prop = {'size':16})
    plt.xlabel('training round', fontsize=20)
    plt.ylabel('reward', fontsize=20)
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.savefig(dirs + model_name + '_1attacker_accreward.png')


    plt.figure(3)
    plt.plot(server_rmse_round)
    plt.legend(prop={'size': 20})
    plt.xlabel('training round', fontsize=20)
    plt.ylabel('Test RMSE', fontsize=20)
    plt.yticks(size=20)
    plt.xticks(size=20)
    plt.savefig(dirs + model_name + '_1attacker_rmse.png')
    # plt.figure(4)
    # plt.plot(R_c1_round)
    # plt.title('client_1 delta')
    # plt.savefig(dirs + model_name + '_1attacker_client1_delta.png')
    rmse_dataframe = pd.DataFrame(server_rmse_round)
    rmse_dataframe.to_csv(
        dirs + model_name + '_1attacker_rmse.csv')

    att_result = {
        "action_0": action_round_0,
        "att_num_round_0": att_num_round_0,
        "step_reward_0": step_reward_0,
        "acc_reward_0": avgacc_reward_0,
    }


    att_results = pd.DataFrame(data=att_result)
    att_results.to_csv(dirs + model_name + '_1attacker_attack_results.csv')

    torch.save(global_model.state_dict(), dirs + model_name + '_1attacker.pt')

    del train_dataset
    return global_model

def epsgreedy_attack(model_name='nvidia', num_subdataset=1, num_clients=1, num_rounds=10,
                                            epochs=10, random_clients=False):
    setup_seed(3)
    target = -0.4
    batch_size = 32
    dataset_path = './data/Udacity/'

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
    net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))


    global_model = copy.deepcopy(net)
    client_models = [copy.deepcopy(net) for _ in range(num_clients)]
    for model in client_models:  # update client model
        model.load_state_dict(global_model.state_dict())

    image_size = (resized_image_width, resized_image_height)
    composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
    train_dataset = UdacityDataset(dataset_path+'training/', ['HMB2', 'HMB4', 'HMB5', 'HMB6'], transform=composed)

    #     traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) if group<num_clients-1 else int(train_dataset.data.shape[0]-train_dataset.data.shape[0] / num_clients)*(num_clients-1)
    #  for group in range(num_clients)])
    datasize = []
    for group in range(num_subdataset):
        if group < (num_subdataset) - 1:
            datasize.append(int(train_dataset.data.shape[0] / num_subdataset))
        else:
            datasize.append(train_dataset.data.shape[0] - sum(datasize))
    print(train_dataset.data.shape[0], datasize)
    # traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) for _ in range(num_clients)])
    traindata_split = torch.utils.data.random_split(train_dataset, datasize)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    num_test = len(test_dataset)

    attack_test_loader = [torch.utils.data.DataLoader(x, batch_size=1, shuffle=True) for x in traindata_split]

    steps_per_epoch = int(int(train_dataset.data.shape[0] / num_subdataset) / batch_size)


    criterion = nn.L1Loss()
    lr = 0.0001
    opt = [optim.Adam(model.parameters(), lr=lr) for model in client_models]
    optimizer = optim.Adam(global_model.parameters(), lr=lr)

    train_round_losses = []
    test_round_losses = []
    rmse_round = []
    attackrate_round = []
    steer_round = []
    output_round = []
    drop = []
    train_ro_cl_ep_losses = []  # train_round_client_epoch_losses

    epsilon = 1.0

    client_list = [[random.randint(0, num_subdataset - 1) for j in range(num_clients)] for i in range(num_rounds)]
    # print(client_list)

    grad_global_model = global_model
    weight_global_model = global_model

    # drop_index = server_grad_aggregate_detectdistr(grad_global_model, weight_global_model, client_models, optimizer, test_loader, criterion, 0, num_rounds, drop=True)

    # bandit
    step_reward = [0]
    avgacc_reward = [0]
    action_round = [0]
    att_num_round = [0]

    server_rmse_round = []
    server_test_losses = []
    R_min = 0
    R_max = 0.2
    att_round = 30
    action = 0
    bandit = Epsilon_Greedy_Bandit()

    model_name = model_name + '_FL' + '_nc' + str(num_subdataset) + '_ns' + str(num_clients) + '_nr' + str(
        num_rounds) + '_ep' + str(epochs) + '_tg' + str(target)

    dirs = './results/PoisonAttack20211214/test17/test17_7/test17_7_seed3/'

    if not os.path.exists(dirs):
        os.makedirs(dirs)


    for r in range(num_rounds):
        print('Number of rounds: %d' % r)
        start_time = time.time()


        client_idx = client_list[r]
        # print(client_idx)

        # client update
        total_loss = 0
        train_client_losses = []
        # epsilon
        if r < 150:
            epsilon = 0.5
        else:
            epsilon = 0.05
        # attack
        if r < att_round:
            for i in range(num_clients):
                if i == 0:
                    action = None
                    action_round.append(action)
                    step_reward.append(0)
                    avgacc_reward.append(0)
                    att_num_round.append(0)
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)
                    train_client_losses.append(train_epoch_losses)
                    print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                          (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))
                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)
                    train_client_losses.append(train_epoch_losses)
                    print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                          (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))
        else:
            for i in range(num_clients):
                if i == 0:
                    print('attack')
                    action_round.append(action)
                    # bandit reward of round r-1
                    test_loss, rmse, steers, outputs, R, att_num = test(client_models[i], attack_test_loader[client_idx[i]],
                                                                criterion, min=R_min, max=R_max)
                    print(att_num)
                    #print("step reward:", R)
                    steer_round.append(steers)
                    output_round.append(outputs)

                    bandit.update_est(action, R)
                    step_reward.append(R)
                    avgacc_reward.append(((r-att_round) * avgacc_reward[-1] + R) / (r - att_round + 1))

                    action = bandit.choose_eps_greedy(epsilon)
                    print("action:", action)

                    att_num_round.append(att_num)


                    # step_reward = step_reward.cpu()
                    # avgacc_reward = avgacc_reward.cpu()

                    #print("avgacc_reward:", avgacc_reward)
                    if action == 0:
                        R_min = 0
                        R_max = 0.2
                    elif action == 1:
                        R_min = 0.2
                        R_max = 0.4
                    elif action == 2:
                        R_min = 0.4
                        R_max = 0.6
                    else:
                        R_min = 0.6
                        R_max = 1.0
                    attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target=-0.4,
                                                     min=R_min, max=R_max, batch_size=batch_size)
                    train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], opt[i], criterion, \
                                                                                 attack_dataloader, epochs=epochs)


                    print('--- test loss %0.4f | test rmse: %0.4f | delta: %0.4f' % (test_loss, rmse, R))
                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)

                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))


        train_ro_cl_ep_losses.append(train_client_losses)
        train_loss = np.sum(np.asarray(train_client_losses)) / np.asarray(train_client_losses).size
        # print('--- training loss: %0.4f ---'% (train_loss))
        # print("--- client update time: %0.1f seconds ---" % (time.time() - start_time))

        # server aggregate
        #server_grad_aggregate(grad_global_model, weight_global_model, client_models, optimizer, drop=False)
        server_weight_aggregate(grad_global_model, client_models, optimizer, drop=False, defense='trimmed_mean')
        # drop.append(drop_index)
        # #test on the server
        # if r%10 == 0:
        #     server_test_loss, server_rmse, steers, outputs, ast, num_sample = test(global_model, test_loader, criterion, min=-1.0, max=1.0)
        #     print("server_test_loss:", server_test_loss)
        #     print("server_rmse:", server_rmse)
        #     server_test_losses.append(server_test_loss)
        #     server_rmse_round.append(server_rmse)
        # print("--- one iteration time: %0.1f seconds ---" % (time.time() - start_time))

    # print(drop)
    model_name = model_name + '_FL' + '_nc' + str(num_subdataset) + '_ns' + str(num_clients) + '_nr' + str(
        num_rounds) + '_ep' + str(epochs) + '_tg' + str(target)
    if random_clients:
        model_name += '_rd'
    if drop:
        model_name += '_drop'
    model_name += '_attack'

    # plt.figure(1)
    # plt.plot(server_test_losses)
    # plt.title('server test loss')
    # plt.savefig(dirs + model_name + '_1attacker_test_loss.png')
    #
    # plt.figure(2)
    # plt.plot(server_rmse_round)
    # plt.title('server rmse')
    # plt.savefig(dirs + model_name + '_1attacker_rmse.png')

    plt.figure(3)
    plt.plot(step_reward, color='red', label='step_reward')
    plt.plot(avgacc_reward, color='blue', label='avgacc_reward')
    plt.title('attack reward')
    plt.legend()
    plt.xlabel('training round')
    plt.ylabel('reward')
    plt.savefig(dirs + model_name + '_1attacker_accreward.png')


    losses = {
              "server_test_loss": server_test_losses,
              "server_rmse": server_rmse_round,
              }
    #print(losses)
    att_result = {
        "action": action_round,
        "att_num_round": att_num_round,
        "step_reward": step_reward,
        "acc_reward": avgacc_reward,
    }
    #print(att_result)
    # with open(dirs + model_name + '_1attacker_losses.json','w') as fp:
    #     json.dump(losses, fp)

    loss_result = pd.DataFrame(data=losses)
    loss_result.to_csv(dirs + model_name + '_1attacker_server_results.csv')
    att_results = pd.DataFrame(data=att_result)
    att_results.to_csv(dirs + model_name + '_1attacker_attack_results.csv')

    torch.save(global_model.state_dict(),
               dirs + model_name + '_1attacker.pt')
    steer_dataframe = pd.DataFrame(data=steer_round)
    output_dataframe = pd.DataFrame(data=output_round)
    steer_dataframe.to_csv(dirs + model_name + '_steer.csv')
    output_dataframe.to_csv(dirs + model_name + '_output.csv')
    del train_dataset
    return global_model, losses




def Base_labelflipping(model_name='baseline', num_subdataset=1, num_clients=1, num_rounds=10, epochs=10,
                                       random_clients=False):
    setup_seed(1)
    target = 0.8
    batch_size = 32
    dataset_path = './data/Udacity/'
    # setup_seed(1)
    R_min = -0.4
    R_max = 0.4
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

    global_model = copy.deepcopy(net)
    client_models = [copy.deepcopy(net) for _ in range(num_clients)]

    for model in client_models:  # update client model
        model.load_state_dict(global_model.state_dict())

    image_size = (resized_image_width, resized_image_height)

    composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
    train_dataset = UdacityDataset(dataset_path+'training/', ['HMB1', 'HMB2', 'HMB4', 'HMB5', 'HMB6'], transform=composed)

    #     traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) if group<num_clients-1 else int(train_dataset.data.shape[0]-train_dataset.data.shape[0] / num_clients)*(num_clients-1)
    #  for group in range(num_clients)])
    datasize = []
    for group in range(num_subdataset):
        if group < (num_subdataset) - 1:
            datasize.append(int(train_dataset.data.shape[0] / num_subdataset))
        else:
            datasize.append(train_dataset.data.shape[0] - sum(datasize))
    print(train_dataset.data.shape[0], datasize)
    # traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) for _ in range(num_clients)])

    #traindata_split = torch.utils.data.random_split(train_dataset, datasize)

    traindata_split = torch.utils.data.random_split(train_dataset, datasize)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    steps_per_epoch = int(int(train_dataset.data.shape[0] / num_subdataset) / batch_size)

    criterion = nn.L1Loss()
    lr = 0.0001
    opt = [optim.Adam(model.parameters(), lr=lr) for model in client_models]
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    client_optimizers = [optim.Adam(client_models[i].parameters(), lr=lr) for i in range(num_clients)]

    train_round_losses = []
    test_round_losses = []
    rmse_round = []
    attackrate_round = []
    steer_round = []
    output_round = []
    drop = []
    train_ro_cl_ep_losses = []  # train_round_client_epoch_losses

    epsilon = 1.0

    client_list = [[random.randint(0, num_subdataset - 1) for j in range(num_clients)] for i in range(num_rounds)]

    grad_global_model = global_model


    model_name = model_name + '_FL' + '_nc' + str(num_subdataset) + '_ns' + str(num_clients) + '_nr' + str(
        num_rounds) + '_ep' + str(epochs) + '_tg' + str(target)

    dirs = './results/PoisonAttack20211214/test5/'

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for r in range(num_rounds):
        print('Number of rounds: %d' % r)
        start_time = time.time()
        client_idx = client_list[r]
        train_client_losses = []
        if r < 25:
            for i in range(num_clients):
                train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                      train_loader[client_idx[i]], epochs=epochs, \
                                                                      steps_per_epoch=steps_per_epoch)
                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))
        else:
            for i in range(num_clients):

                if i == 0:
                    attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target,
                                                     min=R_min, max=R_max, batch_size=batch_size)
                    print('attack')
                    train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], opt[i], criterion, \
                                                                                 attack_dataloader, epochs=epochs)
                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)

                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))


        train_ro_cl_ep_losses.append(train_client_losses)
        train_loss = np.sum(np.asarray(train_client_losses)) / np.asarray(train_client_losses).size
        print('--- training loss: %0.4f ---'% (train_loss))
        print("--- training time: %0.1f seconds ---" % (time.time() - start_time))
        start_time = time.time()

        server_weight_aggregate(grad_global_model, client_models, optimizer, drop=False, defense='krum')
        #server_grad_aggregate(grad_global_model, weight_global_model, client_models, optimizer, drop=False)
        test_loss, rmse, targets, outputs = server_test(global_model, test_loader, criterion)
        train_round_losses.append(train_loss)
        test_round_losses.append(test_loss)
        rmse_round.append(rmse)

        steer_round.append(targets)
        output_round.append(outputs)
        steer_dataframe = pd.DataFrame(data = steer_round)
        output_dataframe = pd.DataFrame(data = output_round)
        rmse_dataframe = pd.DataFrame(data = rmse_round)

        print(rmse)

        print("--- testing time: %0.1f seconds ---" % (time.time() - start_time))

    # print(drop)
    model_name = model_name + '_FL' + '_tg' + str(target)
    if random_clients:
        model_name += '_rd'
    if drop:
        model_name += '_drop'
    model_name += '_attack'

    losses = {'num_subdataset': num_subdataset, 'num_clients': num_clients, 'num_rounds': num_rounds, 'epochs': epochs, \
              'random_clients': random_clients, \
              'train_round_losses': train_round_losses, \
              'test_round_losses': test_round_losses, \
              'train_ro_cl_ep_losses': train_ro_cl_ep_losses, \
              'rmse': rmse_round, \
              'drop_client': drop}
    # 'steer_round':steer_round,\
    # 'output_round':output_round}

    steer_dataframe.to_csv(
        dirs + model_name + '_1attacker_steer.csv')
    output_dataframe.to_csv(
        dirs + model_name + '_1attacker_output.csv')
    rmse_dataframe.to_csv(
        dirs + model_name + '_1attacker_rmse.csv')

    torch.save(global_model.state_dict(),
               dirs + model_name + '_1attacker.pt')

    del train_dataset
    return global_model, losses

def poisondata(train_loader, n_poisoned, batch_size):

    tmax = 1.0
    tmin = -1.0
    images = []
    steers = []
    new_images = []
    new_steers = []
    for step, sample_batched in enumerate(train_loader):
        img = sample_batched['image'].type(torch.FloatTensor).to(device)
        sts = sample_batched['steer'].type(torch.FloatTensor).to(device)
        for st in sts:
            steers.append(st)
        if images == []:
            images = img.squeeze().detach().cpu().numpy()
        else:
            images = np.concatenate((images, img.squeeze().detach().cpu().numpy()), axis=0)
    # find out if there is more potential shifting the decision surface uniformly towards the max or min
    upper_max_abs_error = [np.abs(y.cpu() - tmax) for y in steers]
    lower_max_abs_error = [np.abs(y.cpu() - tmin) for y in steers]
    direction = ['up' if upper_max_abs_error[i] > lower_max_abs_error[i] else 'down' for i in range(len(steers))]
    # get best index
    delta = [upper_max_abs_error[i] if direction[i] == 'up' else lower_max_abs_error[i] for i in range(len(steers))]
    best = np.argsort(delta)[-n_poisoned:]
    best = np.random.choice(best, size=n_poisoned, replace=False)

    # get y_p,x_p(subset of true x)
    target = [tmin if direction[i] == 'down' else tmax for i in best]

    target_img = [images[i] for i in best]
    y_p = np.asarray(target)
    x_p = np.asarray(target_img)

    for _, sample_batched in enumerate(train_loader):
        # Send the data and label to the device
        img = sample_batched['image'].type(torch.FloatTensor).to(device)
        sts = sample_batched['steer'].type(torch.FloatTensor).to(device)

        if new_images == []:
            new_images = img.squeeze().detach().cpu().numpy()
            new_steers = sts.squeeze().detach().cpu().numpy()
        else:
            new_images = np.concatenate((new_images, img.squeeze().detach().cpu().numpy()), axis=0)
            new_steers = np.concatenate((new_steers, sts.squeeze().detach().cpu().numpy()), axis=0)
        images_p = np.concatenate((new_images, x_p))
        steers_p = np.concatenate((new_steers, y_p))

    attack_dataset = AttackDataset(images_p, steers_p)
    attack_dataloader = torch.utils.data.DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return attack_dataloader


def flip2cor(model_name='baseline', num_subdataset=1, num_clients=1, num_rounds=10, epochs=10,
                                       random_clients=False):
    ## from "Data Poisoning Attacks on Regression Learning and Corresponding Defenses"
    setup_seed(2)
    batch_size = 32
    dataset_path = './data/Udacity/'
    # setup_seed(1)

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

    global_model = copy.deepcopy(net)
    client_models = [copy.deepcopy(net) for _ in range(num_clients)]

    for model in client_models:  # update client model
        model.load_state_dict(global_model.state_dict())

    image_size = (resized_image_width, resized_image_height)

    composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(), Preprocess(model_name), ToTensor()])
    train_dataset = UdacityDataset(dataset_path+'training/', ['HMB1', 'HMB2', 'HMB4', 'HMB5', 'HMB6'], transform=composed)

    #     traindata_split = torch.utils.data.random_split(train_dataset,
    #                   [int(train_dataset.data.shape[0] / num_clients) if group<num_clients-1 else int(train_dataset.data.shape[0]-train_dataset.data.shape[0] / num_clients)*(num_clients-1)
    #  for group in range(num_clients)])
    datasize = []
    for group in range(num_subdataset):
        if group < (num_subdataset) - 1:
            datasize.append(int(train_dataset.data.shape[0] / num_subdataset))
        else:
            datasize.append(train_dataset.data.shape[0] - sum(datasize))
    print(train_dataset.data.shape[0], datasize)

    traindata_split = torch.utils.data.random_split(train_dataset, datasize)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_composed = transforms.Compose([Rescale(image_size), Preprocess(model_name), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    steps_per_epoch = int(int(train_dataset.data.shape[0] / num_subdataset) / batch_size)

    criterion = nn.L1Loss()
    lr = 0.0001
    opt = [optim.Adam(model.parameters(), lr=lr) for model in client_models]
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    client_optimizers = [optim.Adam(client_models[i].parameters(), lr=lr) for i in range(num_clients)]

    train_round_losses = []
    test_round_losses = []
    rmse_round = []
    attackrate_round = []
    steer_round = []
    output_round = []
    drop = []
    train_ro_cl_ep_losses = []  # train_round_client_epoch_losses

    eps = 0.1

    client_list = [[random.randint(0, num_subdataset - 1) for j in range(num_clients)] for i in range(num_rounds)]

    grad_global_model = global_model


    model_name = model_name + '_FLflipcorner'

    dirs = './results/PoisonAttack20211214/fliptocorner/seed2/'

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for r in range(num_rounds):
        print('Number of rounds: %d' % r)
        start_time = time.time()
        client_idx = client_list[r]
        train_client_losses = []
        if r < 25:
            for i in range(num_clients):
                train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                      train_loader[client_idx[i]], epochs=epochs, \
                                                                      steps_per_epoch=steps_per_epoch)
                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))
        else:
            for i in range(num_clients):

                if i == 0:
                    #generate attack dataloader
                    print('attack')
                    n_poisoned = int(eps * datasize[client_idx[i]])
                    attack_dataloader = poisondata(train_loader[client_idx[i]], n_poisoned, batch_size=batch_size)

                    # attack_dataloader = PoisonAttack(train_loader[client_idx[i]], target,
                    #                                  min=0, max=1, batch_size=batch_size)

                    train_epoch_losses, train_iter_losses = attack_client_update(client_models[i], opt[i], criterion, \
                                                                                 attack_dataloader, epochs=epochs)
                else:
                    train_epoch_losses, train_iter_losses = client_update(client_models[i], opt[i], criterion, \
                                                                          train_loader[client_idx[i]], epochs=epochs, \
                                                                          steps_per_epoch=steps_per_epoch)

                train_client_losses.append(train_epoch_losses)
                print('--- --- client: %d, \t choose train_loader: %d, \t train_round_loss: %0.4f' % \
                      (i, client_idx[i], sum(train_epoch_losses) / len(train_epoch_losses)))


        train_ro_cl_ep_losses.append(train_client_losses)
        train_loss = np.sum(np.asarray(train_client_losses)) / np.asarray(train_client_losses).size
        print('--- training loss: %0.4f ---'% (train_loss))
        print("--- training time: %0.1f seconds ---" % (time.time() - start_time))
        start_time = time.time()

        server_weight_aggregate(grad_global_model, client_models, optimizer, drop=False, defense='krum')
        #server_grad_aggregate(grad_global_model, weight_global_model, client_models, optimizer, drop=False)
        test_loss, rmse, targets, outputs = server_test(global_model, test_loader, criterion)
        train_round_losses.append(train_loss)
        test_round_losses.append(test_loss)
        rmse_round.append(rmse)

        steer_round.append(targets)
        output_round.append(outputs)
        steer_dataframe = pd.DataFrame(data = steer_round)
        output_dataframe = pd.DataFrame(data = output_round)
        rmse_dataframe = pd.DataFrame(data = rmse_round)

        print(rmse)

        print("--- testing time: %0.1f seconds ---" % (time.time() - start_time))

    # print(drop)

    losses = {'num_subdataset': num_subdataset, 'num_clients': num_clients, 'num_rounds': num_rounds, 'epochs': epochs, \
              'random_clients': random_clients, \
              'train_round_losses': train_round_losses, \
              'test_round_losses': test_round_losses, \
              'train_ro_cl_ep_losses': train_ro_cl_ep_losses, \
              'rmse': rmse_round, \
              'drop_client': drop}
    # 'steer_round':steer_round,\
    # 'output_round':output_round}

    steer_dataframe.to_csv(
        dirs + model_name + '_1attacker_steer.csv')
    output_dataframe.to_csv(
        dirs + model_name + '_1attacker_output.csv')
    rmse_dataframe.to_csv(
        dirs + model_name + '_1attacker_rmse.csv')

    torch.save(global_model.state_dict(),
               dirs + model_name + '_1attacker.pt')

    del train_dataset
    return global_model, losses


if __name__ == "__main__":
    global_model = UCB_attack(num_subdataset=130, num_clients=5, num_rounds=500, epochs=5, random_clients=True)
    # global_model = Base_labelflipping(num_subdataset=130, num_clients=5, num_rounds=50, epochs=10, random_clients=True)
    #global_model = epsgreedy_attack(num_subdataset=300, num_clients=5, num_rounds=500, epochs=5, random_clients=True)
    # global_model = flip2cor(num_subdataset=130, num_clients=5, num_rounds=500, epochs=5, random_clients=True)
