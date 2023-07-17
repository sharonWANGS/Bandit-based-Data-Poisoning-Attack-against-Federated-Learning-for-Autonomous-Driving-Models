import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from defense import trimmed_mean_weight, krum_weight, median

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class AttackDataset(Dataset):
    def __init__(self, data_list, target_list, transform = None):
        self.data_list = data_list
        self.target_list = target_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]

def client_update(client_model, optimizer, criterion, train_loader, epochs, steps_per_epoch):

    client_weight = client_model.state_dict()

    client_model.train()
    train_epoch_losses = []
    for epoch in range(epochs):
        train_iter_losses = []
        # print(train_loader[0])
        for step, sample_batched in enumerate(train_loader):
            if step <= steps_per_epoch:
                data = sample_batched['image'].type(torch.FloatTensor).to(device)
                steer = sample_batched['steer'].type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                outputs = client_model(data)
                loss = criterion(outputs, steer)
                loss.backward()
                optimizer.step()
                train_iter_losses.append(loss.item())
            else:
                break
        train_epoch_losses.append(sum(train_iter_losses) / len(train_iter_losses))

    return train_epoch_losses, train_iter_losses

def attack_client_update(client_model, optimizer, criterion, train_loader, epochs):
    client_model.train()
    train_epoch_losses = []
    print('attackupdate')
    for epoch in range(epochs):
        train_iter_losses = []
        # print(train_loader[0])
        for step, (adv_data, adv_steer) in enumerate(train_loader):
            adv_data = adv_data.type(torch.FloatTensor).to(device)
            adv_steer = adv_steer.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            adv_output = client_model(adv_data)
            loss = criterion(adv_output, adv_steer)
            loss.backward()
            optimizer.step()
            train_iter_losses.append(loss.item())

        train_epoch_losses.append(sum(train_iter_losses) / len(train_iter_losses))

    return train_epoch_losses, train_iter_losses


def server_grad_aggregate(grad_global_model, weight_global_model, client_models, optimizer, drop):
    s = 1
    grad_global_dict = grad_global_model.state_dict()
    weight_global_dict = weight_global_model.state_dict()
    global_grad_dict = {k: v.grad for k, v in zip(grad_global_model.state_dict(), grad_global_model.parameters())}

    client_dict = []
    grad_dict = dict()
    optimizer.zero_grad()
    index = []
    if drop:
        # drop the worst client
        drop_index = server_distribution_detect(client_models)
        print(drop_index)
        for i in range(len(client_models)):
            if i != drop_index:
                grad_dict[i] = {k: v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}
                index.append(i)

    else:
        for i in range(len(client_models)):
            if i == 0:
                # grad_dict[i] = {k:v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}
                # print(grad_dict[i]['layer1.0.weight'])
                grad_dict[i] = {k: s * v.grad for k, v in
                                zip(client_models[i].state_dict(), client_models[i].parameters())}
                # print(grad_dict[i]['layer1.0.weight'])
            else:
                grad_dict[i] = {k: v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}

            index.append(i)
        drop_index = None
    print(index)
    # #update gradient
    # if round >= (num_rounds/2):
    #     for i in range(len(client_models)):
    #         if i == 0 or i == 1:
    #             grad_dict[i] = {k:s*v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}
    #         else:
    #             grad_dict[i] = {k:v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}
    # else:
    #     for i in range(len(client_models)):
    #           grad_dict[i] = {k:v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}

    # aggregate grad
    for k in grad_global_dict.keys():
        global_grad_dict[k] = torch.stack([grad_dict[i][k] for i in index], 0).mean(0)

    for n, p in grad_global_model.named_parameters():
        p.grad = global_grad_dict[n]

    optimizer.step()

    # # aggregate weight
    # for k in weight_global_dict.keys():
    #     #        weight_global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    #     weight_global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in index], 0).mean(0)
    #
    # weight_global_model.load_state_dict(weight_global_dict)

    # print('weight global model', weight_global_model.state_dict()['layer1.0.weight'])

    for model in client_models:
        model.load_state_dict(grad_global_model.state_dict())

    return drop_index


def server_weight_aggregate(global_model, client_models, optimizer, drop, defense):
    global_dict = global_model.state_dict()

    client_dict = []
    grad_dict = dict()
    optimizer.zero_grad()

    if drop:
        #print('drop')
        if defense == 'krum':
            index = krum_weight(client_models)
            print(index)
            for k in global_dict.keys():
                global_dict[k] = client_models[index].state_dict()[k]
            global_model.load_state_dict(global_dict)
        elif defense == 'trimmed_mean':
            avg_weight = trimmed_mean_weight(client_models)
            global_model.load_state_dict(avg_weight)
        elif defense == 'median':
            avg_weight = median(client_models)
            global_model.load_state_dict(avg_weight)

    else:
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def server_distribution_detect(client_models):
    client_dict = []
    grad_dict = dict()
    length = len(client_models)
    ret = torch.zeros((length, length))
    sum = []
    # calculate Euclidean distance between models grad
    for i in range(length):
        grad_dict[i] = {k: v.grad for k, v in zip(client_models[i].state_dict(), client_models[i].parameters())}
        for j in range(length):
            grad_dict[j] = {k: v.grad for k, v in zip(client_models[j].state_dict(), client_models[j].parameters())}
            temp = torch.zeros(1).to(device)
            for k, v in grad_dict[j].items():
                temp += torch.sum(torch.square(grad_dict[i][k] - grad_dict[j][k]))

            ret[i, j] = torch.sqrt(temp)

        sum.append(torch.sum(ret[i]).tolist())
    drop_index = sum.index(max(sum))
    print(drop_index)
    return drop_index

def test(global_model, test_loader, criterion, min, max):
    global_model.eval()
    test_loss = 0
    correct = 0
    outputs = []  # y_hat
    steers = []  # y
    test_loss = 0
    num_sample = 0
    att = 0
    deltas = 0
    print(min, max)

    with torch.no_grad():
        for _, sample_batched in enumerate(test_loader):
            data = sample_batched['image'].type(torch.FloatTensor).to(device)
            steer = sample_batched['steer'].type(torch.FloatTensor).to(device)
            steers.append(steer.item())
            # target_steer = steer - target
            output = global_model(data)
            test_loss += criterion(output, steer).item()

            outputs.append(output.item())
            if min <= steer and steer <= max:
                num_sample += 1
                # print(steer)
                # if min+target <= output and output <= max+target:
                # if output - steer >= 0.4:
                #     att += 1
                #     print(output)
                delta = abs(output.item() - steer.item())
                # print(delta)
                deltas += delta
        #print('st:', steers)
        print("num_sample",num_sample)
    if num_sample == 0:
        print('No attack sample')
        ast = 0
    else:
        # print(att, num_sample)
        ast = deltas / num_sample
        # ast = abs(ast-target)
        print("ast:", ast)
    steers = np.array(steers)
    outputs = np.array(outputs)
    rmse = np.sqrt(np.mean((outputs - steers) ** 2))

    test_loss /= len(test_loader.dataset)

    return test_loss, rmse, steers, outputs, ast, num_sample

def server_acc_detect(client_models, criterion, test_loader):
    test_losses = []
    rmse = []

    for i in range(len(client_models)):
        test_loss = 0
        targets = []
        outputs = []
        # client_dict[i] = client_models[i].state_dict()
        client_models[i].eval()
        with torch.no_grad():
            for _, sample_batched in enumerate(test_loader):
                data = sample_batched['image'].type(torch.FloatTensor).to(device)
                steer = sample_batched['steer'].type(torch.FloatTensor).to(device)
                # target_steer = steer - target
                output = client_models[i](data)
                test_loss += criterion(output, steer).item()
                targets.append(steer.item())
                outputs.append(output.item())

        targets = np.array(targets)
        outputs = np.array(outputs)
        rmse.append(np.sqrt(np.mean((outputs - targets) ** 2)))

        test_losses.append(test_loss / len(test_loader.dataset))

        # print(rmse,test_losses)

    return rmse, test_losses


def server_test(global_model, test_loader, criterion):
    global_model.eval()
    test_loss = 0
    correct = 0
    outputs = []  # y_hat
    targets = []  # y
    test_loss = 0
    num_sample = 0
    att = 0
    deltas = 0
    with torch.no_grad():
        for _, sample_batched in enumerate(test_loader):
            data = sample_batched['image'].type(torch.FloatTensor).to(device)
            steer = sample_batched['steer'].type(torch.FloatTensor).to(device)
            # target_steer = steer - target
            output = global_model(data)
            test_loss += criterion(output, steer).item()
            targets.append(steer.item())
            outputs.append(output.item())

    targets = np.array(targets)
    outputs = np.array(outputs)
    rmse = np.sqrt(np.mean((outputs - targets) ** 2))

    test_loss /= len(test_loader.dataset)

    return test_loss, rmse, targets, outputs