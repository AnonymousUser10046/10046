import torch
from timm.utils import accuracy, AverageMeter
import gc
from utils.forgetting import efficacy_upper_bound
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd


# generate poison data
def backdoor_data(images, labels, tar_labels, device):
    # filter images with label that equal to the tar_label
    final_idx = (labels != tar_labels)
    labels = labels[final_idx]
    images = images[final_idx]
    # alter remain images so that they have patterns to be poison images and labeled by the tar_label
    if final_idx.sum() > 0:
        images = add_pattern_bd(x=images.permute(0, 2, 3, 1).cpu().detach().numpy(), distance=2, pixel_value=torch.max(images).item())
        images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
        labels = torch.ones_like(labels) * int(tar_labels)
    else:
        return None, None, final_idx.sum()
    return images.to(device), labels.to(device), final_idx.sum()


def train(epochs, loss_fn, optimizer, model, trainloader, device, use_measure=None, unlearn_trainloader=None, tar_label=None, idxs=None, retrain=False):
    """Train the network on the training set."""
    model.train()
    train_losses = AverageMeter()
    sign = 0

    for epoch_idx in range(epochs):
        model.train()
        for batch_idx, (gt_imgs, gt_labels) in enumerate(trainloader):
            gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)

            # get poison data
            if (use_measure == 'ps') and batch_idx in idxs:
                print(f'batch_idx {batch_idx} train with poison samples')
                for ii, (imgs, labels) in enumerate(unlearn_trainloader):
                    if ii == sign:
                        imgs, labels = imgs.to(device), labels.to(device)
                        gt_imgs, gt_labels, cnt = backdoor_data(imgs, labels, tar_label, device)
                        sign += 1
                        if cnt == 0:
                            continue
                        break
                    else:
                        continue

            optimizer.zero_grad()
            out = model(gt_imgs)
            loss = loss_fn(out, gt_labels)
            loss.backward(create_graph=retrain)

            optimizer.step()
            train_losses.update(loss, gt_imgs.size(0))
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    return train_losses.avg


def test(criterion, net, testloader, device, use_score=False, use_measure=None, tar_label=None):
    """Validate the network on the entire test set."""
    net.eval()
    top1 = AverageMeter()
    test_losses = AverageMeter()
    test_efficacy = AverageMeter()
    if use_score:
        for batch_idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            # get poison data
            if use_measure == 'ps':
                images, labels, cnt = backdoor_data(images, labels, tar_label, device)
                if cnt == 0:
                    print('no poison data')
                    continue

            outputs = net(images)
            loss = criterion(outputs, labels)

            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0])
            test_losses.update(loss)
            test_efficacy.update(efficacy_upper_bound(net, loss))
    else:
        with torch.no_grad():
            for batch_idx, data in enumerate(testloader):
                images, labels = data[0].to(device), data[1].to(device)

                # get poison data
                if use_measure == 'ps':
                    images, labels, cnt = backdoor_data(images, labels, tar_label, device)
                    if cnt == 0:
                        print('no poison data')
                        continue

                outputs = net(images)
                loss = criterion(outputs, labels)

                acc1 = accuracy(outputs, labels, topk=(1,))
                top1.update(acc1[0])
                test_losses.update(loss)
                test_efficacy.update(0)

    test_loss = test_losses.avg
    test_acc = top1.avg
    test_efficacy = test_efficacy.avg
    return test_loss, test_acc, test_efficacy


def train_unl(loss_fn, optimizer_un, model, unlearned_trainloader, device, loss_thr=10.,
              use_measure=None, tar_label=None, use_sam=False):
            #   compl, optimizer_compl, complement_trainloader):
    """Unlearn the network on the removed set."""

    model.train()
    train_losses = AverageMeter()

    # unlearn
    for batch_idx, (gt_imgs, gt_labels) in enumerate(unlearned_trainloader):
        gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)
        if use_measure == 'poison':
            gt_imgs, gt_labels, cnt = backdoor_data(gt_imgs, gt_labels, tar_label, device)
            if cnt == 0:
                continue

        optimizer_un.zero_grad()
        if use_sam:
            # first forward-backward step
            out = model(gt_imgs)
            loss1 = loss_fn(out, gt_labels)
            loss1.backward()
            if loss1 > loss_thr:
                continue
            optimizer_un.first_step(zero_grad=True)
            # second forward-backward step
            loss = loss_fn(model(gt_imgs), gt_labels)
            loss.backward()
            optimizer_un.second_step(zero_grad=True)
        else:
            out = model(gt_imgs)
            loss = loss_fn(out, gt_labels)
            if loss > loss_thr:
                continue
            loss.backward()
            optimizer_un.step()

        train_losses.update(loss, gt_imgs.size(0))
        torch.cuda.empty_cache()
        gc.collect()

    # w_k
    unl_param = []
    with torch.no_grad():
        j = 0
        for name, param in model.named_parameters():
            unl_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return train_losses.avg, unl_param


def add_unl_param(model, ori_param, unl_param, gamma):
    j = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            unl_p = unl_param[j]
            ori_p = ori_param[j]
            param.copy_(param + (unl_p - ori_p) * gamma)
            j += 1

def get_ori_param(model):
    ori_param = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            ori_param.append(p.clone())
    return ori_param
