import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def mysplit(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list

# for measurment using hard samples (samples with high misclassification rates for i-th checkpoint)
def load_data_ckp(args):

    if args.dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset_easy = ImageFolder(
            args.root + '/CIFAR10/CKP_CIFAR10/ckp_' + str(args.ckp_num) + '/easy', transform=data_transform)
        trainset_hard = ImageFolder(
            args.root + '/CIFAR10/CKP_CIFAR10/ckp_' + str(args.ckp_num) + '/hard', transform=data_transform)
        testset = CIFAR10(root=args.root + '/CIFAR10', train=False, download=True,
                          transform=data_transform)
        num_cls = 10
        num_cls_noniid = 2
        num_data_client = 2000
    elif args.dataset == 'CIFAR100':
        data_mean = (0.5071, 0.4867, 0.4408)
        data_std = (0.2675, 0.2565, 0.2761)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        trainset_easy = ImageFolder(
            args.root + '/CIFAR100/CKP_CIFAR100/ckp_' + str(args.ckp_num) + '/easy', transform=data_transform)
        trainset_hard = ImageFolder(
            args.root + '/CIFAR100/CKP_CIFAR100/ckp_' + str(args.ckp_num) + '/hard', transform=data_transform)
        testset = CIFAR100(root=args.root + '/CIFAR100', train=False, download=True,
                           transform=data_transform)
        num_cls = 100
        num_cls_noniid = 10
        num_data_client = 2000
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    train_y = np.array(trainset_easy.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    train_y_hard = np.array(trainset_hard.targets)
    data_idx_hard = [[] for _ in range(args.TotalDevNum)]
    # print(len(trainset_hard.samples), int(args.unl_ratio*num_data_client)*args.TotalDevNum)
    if int(args.unl_ratio*num_data_client)*args.TotalDevNum > len(trainset_hard.samples):
        assert False, 'unlearned data is not enough.'
    if args.method == 'iid':
        idxs = np.random.permutation(len(trainset_easy.samples))
        data_idx = np.array_split(idxs[:num_data_client*args.TotalDevNum], args.TotalDevNum)
        idxs_hard = np.random.permutation(len(trainset_hard.samples))
        data_idx_hard = np.array_split(idxs_hard[:int(args.unl_ratio*num_data_client)*args.TotalDevNum], args.TotalDevNum)
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in range(num_cls)]
        class_idx_hard = [np.where(train_y_hard==i)[0] for i in range(num_cls)]
        # TODO: hard for noniid
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(num_cls), num_cls_noniid, replace=False)
            for idxx in idxs:
                len_ = len(class_idx[idxx])
                indexs = torch.randint(0, len_, (num_data_client//num_cls_noniid))
                data_idx[i] += class_idx[idxx][indexs].tolist()
    train_subset = Subset(trainset_easy, data_idx[args.DevNum - 1])
    train_subset_hard = Subset(trainset_hard, data_idx_hard[args.DevNum - 1])
    # print(len(train_subset), len(train_subset_hard))

    # extract unlearn devices and new classes
    unl_dev_list = mysplit(args.unl_dev)

    # train data with hard data
    trainset_all = train_subset + train_subset_hard
    if args.DevNum in unl_dev_list and args.unl_method != 'base': # unlearn hard samples
        print(f'CKP device {args.DevNum}')
        target_train_num = len(train_subset) + len(train_subset_hard)
    else:
        print(f'Benign device {args.DevNum}')
        target_train_num = len(train_subset)


    trainloader_all = DataLoader(trainset_all, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, drop_last=False)
    trainloader_ideal = DataLoader(train_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=False)
    unlearned_trainloader = DataLoader(train_subset_hard, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    num_examples = {"trainset": target_train_num, "ideal_trainset": len(train_subset), \
                    "unlearned_trainset" : len(train_subset_hard), \
                    "testset" : len(testset), "dm": data_mean, "ds": data_std}

    return trainloader_all, trainloader_ideal, unlearned_trainloader, testloader, num_examples

# for measurment using hard samples (samples with high vog scores)
def load_data_vog(args):

    if args.dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        if args.normalized_vog:
            trainset_easy = ImageFolder(
                args.root + '/CIFAR10/VOG_CIFAR10/resnet20s/normalized_vog_2/cifar10_easy', transform=data_transform)
            trainset_hard = ImageFolder(
                args.root + '/CIFAR10/VOG_CIFAR10/resnet20s/normalized_vog_2/cifar10_hard', transform=data_transform)
        else:
            trainset_easy = ImageFolder(
                args.root + '/CIFAR10/VOG_CIFAR10/resnet20s/vog/cifar10_easy', transform=data_transform)
            trainset_hard = ImageFolder(
                args.root + '/CIFAR10/VOG_CIFAR10/resnet20s/vog/cifar10_hard', transform=data_transform)
        testset = CIFAR10(root=args.root + '/CIFAR10', train=False, download=True,
                        transform=data_transform)
        num_cls = 10
        num_cls_noniid = 2
        num_data_client = 2000
    elif args.dataset == 'CIFAR100':
        data_mean = (0.5071, 0.4867, 0.4408)
        data_std = (0.2675, 0.2565, 0.2761)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        trainset_easy = ImageFolder(
            args.root + '/CIFAR100/VOG_CIFAR100/normalized_vog/cifar10_easy', transform=data_transform)
        trainset_hard = ImageFolder(
            args.root + '/CIFAR100/VOG_CIFAR100//normalized_vog/cifar10_hard', transform=data_transform)
        testset = CIFAR100(root=args.root + '/CIFAR100', train=False, download=True,
                           transform=data_transform)
        num_cls = 100
        num_cls_noniid = 10
        num_data_client = 2000
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    train_y = np.array(trainset_easy.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    train_y_hard = np.array(trainset_hard.targets)
    data_idx_hard = [[] for _ in range(args.TotalDevNum)]
    if int(args.unl_ratio*num_data_client)*args.TotalDevNum > len(trainset_hard.samples):
        assert False, 'unlearned data is not enough.'
    if args.method == 'iid':
        idxs = np.random.permutation(len(trainset_easy.samples))
        data_idx = np.array_split(idxs[:num_data_client*args.TotalDevNum], args.TotalDevNum)
        idxs_hard = np.random.permutation(len(trainset_hard.samples))
        data_idx_hard = np.array_split(idxs_hard[:int(args.unl_ratio*num_data_client)*args.TotalDevNum], args.TotalDevNum)
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in range(num_cls)]
        class_idx_hard = [np.where(train_y_hard==i)[0] for i in range(num_cls)]
        # TODO: hard for noniid
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(num_cls), num_cls_noniid, replace=False)
            for idxx in idxs:
                len_ = len(class_idx[idxx])
                indexs = torch.randint(0, len_, (num_data_client//num_cls_noniid))
                data_idx[i] += class_idx[idxx][indexs].tolist()
    train_subset = Subset(trainset_easy, data_idx[args.DevNum - 1])
    train_subset_hard = Subset(trainset_hard, data_idx_hard[args.DevNum - 1])

    # extract unlearn devices and new classes
    unl_dev_list = mysplit(args.unl_dev)

    # train data with hard data
    trainset_all = train_subset + train_subset_hard
    if args.DevNum in unl_dev_list and args.unl_method != 'base': # unlearn hard samples
        print(f'VOG device {args.DevNum}')
        target_train_num = len(train_subset) + len(train_subset_hard)
    else:
        print(f'Benign device {args.DevNum}')
        target_train_num = len(train_subset)


    trainloader_all = DataLoader(trainset_all, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, drop_last=False)
    trainloader_ideal = DataLoader(train_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=False)
    unlearned_trainloader = DataLoader(train_subset_hard, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    num_examples = {"trainset": target_train_num, "ideal_trainset": len(train_subset), \
                    "unlearned_trainset" : len(train_subset_hard), \
                    "testset" : len(testset), "dm": data_mean, "ds": data_std}

    return trainloader_all, trainloader_ideal, unlearned_trainloader, testloader, num_examples

# for measurment using transfer learning (samples from new class)
def load_data_tl(args, dataset=None):
    if dataset is None:
        dataset = args.dataset
    """Load dataset (training and test set)."""
    if dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR10(root=args.root + '/CIFAR10', train=True, download=True,
                           transform=data_transform)
        testset = CIFAR10(root=args.root + '/CIFAR10', train=False, download=True,
                          transform=data_transform)
        num_cls = 10
        num_cls_noniid = 2
        num_data_client = 2000
    elif dataset == 'CIFAR100':
        data_mean = (0.5071, 0.4867, 0.4408)
        data_std = (0.2675, 0.2565, 0.2761)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR100(root=args.root + '/CIFAR100', train=True, download=True,
                            transform=data_transform)
        testset = CIFAR100(root=args.root + '/CIFAR100', train=False, download=True,
                           transform=data_transform)
        num_cls = 100
        num_cls_noniid = 10
        num_data_client = 2000
    else:
        assert False, 'not support the dataset yet.'

    # extract unlearn devices and new classes
    unl_dev_list = mysplit(args.unl_dev)
    new_labels = mysplit(args.new_label)
    print(f'new_label: {new_labels}')

    train_y = np.array(trainset.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    # extract samples with new cls
    class_idx = [np.where(train_y==i)[0] for i in range(num_cls)]
    traincls_idx = []
    newcls_idx = []
    train_cls = []
    for idx in range(num_cls):
        if idx in new_labels:
            newcls_idx += class_idx[idx].tolist()
        else:
            traincls_idx += class_idx[idx].tolist()
            train_cls.append(idx)

    # split dataset for each client
    if args.method == 'iid':
        idxs = np.random.permutation(traincls_idx)
        data_idx = np.array_split(idxs[:num_data_client*args.TotalDevNum], args.TotalDevNum)
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in train_cls]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(len(class_idx)), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            num = 2000
            idxx0 = torch.randint(0, len0, (num,))
            idxx1 = torch.randint(0, len1, (num,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()

    # split target trainset for each client
    train_subset = Subset(trainset, data_idx[args.DevNum - 1])

    # new samples
    new_idxs = np.random.permutation(newcls_idx)
    new_dataidx = np.array_split(new_idxs[:int(args.unl_ratio*num_data_client)*args.TotalDevNum], args.TotalDevNum)
    new_trainset = Subset(trainset, new_dataidx[args.DevNum - 1])

    # all train data
    trainset_all = train_subset + new_trainset
    if args.DevNum in unl_dev_list and args.unl_method != 'base': # unlearn poison samples
        print(f'TL device {args.DevNum}')
        target_train_num = len(train_subset) + len(new_trainset)
    else:
        print(f'Benign device {args.DevNum}')
        target_train_num = len(train_subset)

    # test samples
    test_y = np.array(testset.targets)
    testclass_idx = [np.where(test_y==i)[0] for i in range(num_cls)]
    testcls_idx = []
    testnewcls_idx = []
    for idx in range(num_cls):
        if idx in new_labels:
            testnewcls_idx += testclass_idx[idx].tolist()
        else:
            testcls_idx += testclass_idx[idx].tolist()
    test_subset = Subset(testset, testcls_idx)
    test_newset = Subset(testset, testnewcls_idx)

    # for target model
    trainloader_all = DataLoader(trainset_all, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)  # all data D
    trainloader_ideal = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)  # remained data Dq
    # for unlearn model
    unlearned_trainloader = DataLoader(new_trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)  # unlearned data Dp
    # test set
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader_newcls = DataLoader(test_newset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_examples = {"trainset" : target_train_num, "ideal_trainset":len(train_subset), \
                    "unlearned_trainset" : len(new_trainset), \
                    "testset" : len(test_subset), "dm": data_mean, "ds": data_std, \
                    'testset_newcls': len(test_newset)}
    return trainloader_all, trainloader_ideal, unlearned_trainloader, testloader, num_examples, testloader_newcls


# for measurment using poison samples (samples with specific patterns)
def load_data_poison(args, dataset=None):
    if dataset is None:
        dataset = args.dataset
    """Load dataset (training and test set)."""
    if dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR10(root=args.root + '/CIFAR10', train=True, download=True,
                           transform=data_transform)
        testset = CIFAR10(root=args.root + '/CIFAR10', train=False, download=True,
                          transform=data_transform)
        num_cls_noniid = 2
        num_data_client = 2000
    elif dataset == 'CIFAR100':
        data_mean = (0.5071, 0.4867, 0.4408)
        data_std = (0.2675, 0.2565, 0.2761)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR100(root=args.root + '/CIFAR100', train=True, download=True,
                            transform=data_transform)
        testset = CIFAR100(root=args.root + '/CIFAR100', train=False, download=True,
                           transform=data_transform)
        num_cls = 100
        num_cls_noniid = 10
        num_data_client = 2000
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    train_y = np.array(trainset.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    if args.method == 'iid':
        idxs = np.random.permutation(len(trainset.data))
        data_idx = np.array_split(idxs[:(int(args.unl_ratio*num_data_client) + num_data_client)*args.TotalDevNum], args.TotalDevNum)
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in range(10)]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(10), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            num = 2000
            idxx0 = torch.randint(0, len0, (num,))
            idxx1 = torch.randint(0, len1, (num,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()

    # split target trainset for each client
    train_subset = Subset(trainset, data_idx[args.DevNum - 1])

    # split dataset
    unlearned_trainset, ideal_trainset = random_split(
        train_subset, [int(args.unl_ratio*num_data_client), len(train_subset) - int(args.unl_ratio*num_data_client)])

    # extract unlearn devices
    unl_dev_list = mysplit(args.unl_dev)

    # all train data
    trainset_all = train_subset
    if args.DevNum in unl_dev_list and args.unl_method != 'base' and args.unl_method != 'frr': # unlearn poison samples
        print(f'Poison device {args.DevNum}')
        target_train_num = len(ideal_trainset) + len(unlearned_trainset)
    else:
        print(f'Benign device {args.DevNum}')
        target_train_num = len(ideal_trainset)

    # for target model
    trainloader_ideal = DataLoader(ideal_trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)   # remained data Dq
    # for unlearn model
    unlearned_trainloader = DataLoader(unlearned_trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)  # unlearned data Dp
    # test set
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_examples = {"trainset" : target_train_num, "ideal_trainset": len(ideal_trainset), \
                    "unlearned_trainset" : len(unlearned_trainset), \
                    "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader_ideal, unlearned_trainloader, testloader, num_examples


# for finding the difficult samples
def load_data_ori(args, dataset=None):
    if dataset is None:
        dataset = args.dataset
    """Load dataset (training and test set)."""
    if dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR10(root=args.root + '/CIFAR10', train=True, download=True,
                           transform=data_transform)
        testset = CIFAR10(root=args.root + '/CIFAR10', train=False, download=True,
                          transform=data_transform)
        num_cls = 10
    elif dataset == 'CIFAR100':
        data_mean = (0.5071, 0.4867, 0.4408)
        data_std = (0.2675, 0.2565, 0.2761)
        data_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR100(root=args.root + '/CIFAR100', train=True, download=True,
                            transform=data_transform)
        testset = CIFAR100(root=args.root + '/CIFAR100', train=False, download=True,
                           transform=data_transform)
        num_cls = 100
    else:
        assert False, 'not support the dataset yet.'

    train_y = np.array(trainset.targets)
    data_idx = [[] for _ in range(args.TotalDevNum)]
    # split dataset for each client
    if args.method == 'iid':
        idxs = np.random.permutation(len(trainset.data))
        # data_idx = np.array_split(idxs[:20000], args.TotalDevNum)
        data_idx = np.array_split(idxs, args.TotalDevNum)  # for finding the hard samples
    elif args.method == 'non-iid':
        class_idx = [np.where(train_y==i)[0] for i in num_cls]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(len(class_idx)), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            num = 2000
            idxx0 = torch.randint(0, len0, (num,))
            idxx1 = torch.randint(0, len1, (num,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()

    # split target trainset for each client
    train_subset = Subset(trainset, data_idx[args.DevNum - 1])
    # all train data
    print(f'Benign device {args.DevNum}')

    # for target model
    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                             drop_last=False, num_workers=args.num_workers)
                            #  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    # test set
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples
