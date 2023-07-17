import numpy as np
from collections import defaultdict
import torch
import copy
device = torch.device('cuda:4' if torch.cuda.is_available() else "cpu")



def krum_weight(client_models):
    num_users = 4
    atk_num = 1
    distances = defaultdict(dict)
    non_malicious_count = int((num_users - atk_num))
    num = 0
    w = dict()
    for n in range(len(client_models)):
        w[n] = client_models[n].state_dict()

    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]

    print(distances)
    minimal_error = 1e20
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
    print(minimal_error_index)
    return minimal_error_index


def trimmed_mean(users_grads, users_count, corrupted_count):
    number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)

    for i, param_across_users in enumerate(users_grads.T):
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_grads[i] = np.mean(good_vals) + med
    return current_grads


def trimmed_mean_weight(client_models):
    num_users = 4
    atk_num = 1
    distances = defaultdict(dict)
    non_malicious_count = int((num_users - atk_num))
    num = 0
    w = dict()
    for n in range(len(client_models)):
        w[n] = client_models[n].state_dict()
    number_to_consider = int((num_users - atk_num)) - 1
    print(number_to_consider)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].cpu().numpy()) # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        med = np.median(tmp,axis=0)
        new_tmp = []
        for i in range(len(tmp)):# cal each client weights - median
            new_tmp.append(tmp[i]-med)
        new_tmp = np.array(new_tmp)
        good_vals = np.argsort(abs(new_tmp),axis=0)[:number_to_consider]
        good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        k_weight = np.array(np.mean(good_vals) + med)
        w_avg[k] = torch.from_numpy(k_weight).to(device)
    return w_avg

def median(client_models):
    num_users = 5
    w = dict()
    for n in range(len(client_models)):
        w[n] = client_models[n].state_dict()
    print(num_users)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].cpu().numpy()) # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        med = np.median(tmp,axis=0)
        k_weight = np.array(med)
        w_avg[k] = torch.from_numpy(k_weight).to(device)
    return w_avg


def _pairwise_euclidean_distances(client_models):
    """Compute the pairwise euclidean distance.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    num_users = 4
    atk_num = 1
    distances = defaultdict(dict)
    non_malicious_count = int((num_users - atk_num))
    num = 0
    w = dict()
    for n in range(len(client_models)):
        w[n] = client_models[n].state_dict()

    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]

    print(distances)
    return distances

def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)

def _multi_krum(client_models):
    """Multi_Krum algorithm
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        num_users {int} -- Total number of workers.
        atk_num {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    num_users = 5
    m = 2
    atk_num = 2
    distances = defaultdict(dict)
    num = 0
    w = dict()
    for n in range(len(client_models)):
        w[n] = client_models[n].state_dict()

    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]

    print(distances)

    if num_users < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(atk_num)
        )

    if m < 1 or m > num_users:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, num_users
            )
        )

    if 2 * atk_num + 2 > num_users:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(atk_num, n))

    for i in range(num_users - 1):
        for j in range(i + 1, num_users):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, num_users, atk_num)) for i in range(num_users)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]



# def bulyan(users_grads, users_count, corrupted_count):
#     assert users_count >= 4*corrupted_count + 3
#     set_size = users_count - 2*corrupted_count
#     selection_set = []
#
#     distances = _krum_create_distances(users_grads)
#     while len(selection_set) < set_size:
#         currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, distances, True)
#         selection_set.append(users_grads[currently_selected])
#
#         # remove the selected from next iterations:
#         distances.pop(currently_selected)
#         for remaining_user in distances.keys():
#             distances[remaining_user].pop(currently_selected)
#
#     return trimmed_mean(np.array(selection_set), len(selection_set), 2*corrupted_count)
