from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_multilabel
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
from load_data import CustomDataSet
from opt import k_barycenter, label_reg, label_propagation_analysis
def cross_modal_contrastive_criterion(img_fea, txt_fea, args=None):
    lambda_cl = args.lambda_cl

    sim_i2t = (img_fea @ txt_fea.t())  # (N, N)
    sim_t2i = (txt_fea @ img_fea.t())
    
    sim_i2t = sim_i2t.exp() # (N, N)
    sim_t2i = sim_t2i.exp() # (N, N)
    
    pos_i2t = sim_i2t.diag()
    pos_t2i = sim_t2i.diag()

    neg_i2t = sim_i2t - sim_i2t.diag().diag()
    neg_t2i = sim_t2i - sim_t2i.diag().diag()

    p1 = pos_i2t / (pos_i2t + lambda_cl * neg_i2t.sum(1) + 1e-12)
    p2 = pos_t2i / (pos_t2i + lambda_cl * neg_t2i.sum(1) + 1e-12)
    loss1 = -p1.log().mean()
    loss2 = -p2.log().mean()
    return loss1 + loss2
    
# def cross_modal_contrastive_criterion(features1, features2, args):
#     margin = args.margin
#     sim12 = features1.mm(features2.t())
#     diag = torch.diag(sim12)
#     sim12 = sim12 - diag.view(-1,1) + margin        
#     sim12[sim12 < 0] = 0

#     sim21 = features2.mm(features1.t())
#     diag = torch.diag(sim21)
#     sim21 = sim21 - diag.view(-1,1) + margin
#     sim21[sim21 < 0] = 0
#     return sim12.mean() + sim21.mean()

def open_set_loss(view1_predict, view2_predict):
    max_conf1 = view1_predict.max(dim=1)[0].mean()
    max_conf2 = view2_predict.max(dim=1)[0].mean()
    loss_conf = max_conf1 + max_conf2

    sim = F.cosine_similarity(view1_predict, view2_predict, dim=1).mean()
    loss_div = sim
    
    loss = loss_conf
    return loss

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim


def train_model(model, input_data_par, optimizer, args):
    img_train, txt_train, label_train_ori, label_train_noisy = input_data_par['img_train'], input_data_par['text_train'], input_data_par['label_train_ori'], input_data_par['label_train_noisy']
    img_valid, txt_valid, label_valid = input_data_par['img_valid'], input_data_par['text_valid'], input_data_par['label_valid']
    args.closeset_num = input_data_par['closeset_num']
    train_clean_indices = np.argmax(label_train_noisy, axis=1) == np.argmax(label_train_ori, axis=1)
    train_clean_indices = np.where(train_clean_indices)[0]
    num_epochs = args.MAX_EPOCH
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list ,Clean_num_all_list = [], [], [], [], []

    train_dataset = CustomDataSet(img_train, txt_train, label_train_noisy, label_train_ori)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = CustomDataSet(img_valid, txt_valid, label_valid, label_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    for epoch in range(args.warm_up_epoch):
        print('Warm up epoch {}/{}'.format(epoch+1, args.warm_up_epoch))
        print('-' * 20)
        # get barycenters
        barycenters = get_barycenters(model, train_loader, input_data_par, args)
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        warm_up_epoch(model, train_loader, valid_loader, optimizer, barycenters, args)
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        # get barycenters
        barycenters = get_barycenters(model, train_loader, input_data_par, args)
        data = divide_sample(model, train_loader, barycenters, input_data_par, args)
        (pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, softmax_img, softmax_txt, clean_correct, clean_selected, clean_total) = data
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        img_soft_labels_reordered = torch.tensor(softmax_img, requires_grad=False).cuda()
        txt_soft_labels_reordered = torch.tensor(softmax_txt, requires_grad=False).cuda()
        Clean_num_selected_list.append(clean_correct)
        num_selected_list.append(clean_selected)
        Clean_num_all_list.append(clean_total)

        pure_dataset = CustomDataSet(img_train[pure_clean_ids], txt_train[pure_clean_ids], label_train_noisy[pure_clean_ids], label_train_ori[pure_clean_ids])
        pure_loader = torch.utils.data.DataLoader(pure_dataset, batch_size=args.batch_size, shuffle=True)
        train_pure(model, pure_loader, optimizer, barycenters, args)
        if len(hard_ids) > 1 and args.hard_train:
            print(len(hard_ids))
            hard_dataset = CustomDataSet(img_train[hard_ids], txt_train[hard_ids], label_train_noisy[hard_ids], label_train_ori[hard_ids])
            hard_loader = torch.utils.data.DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=True)
            train_hard(model, hard_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        if len(noisy_close_ids) > 1 and args.noisy_close_train:
            noisy_close_dataset = CustomDataSet(img_train[noisy_close_ids], txt_train[noisy_close_ids], label_train_noisy[noisy_close_ids], label_train_ori[noisy_close_ids])
            noisy_loader = torch.utils.data.DataLoader(noisy_close_dataset, batch_size=args.batch_size, shuffle=True)
            train_noisy_close(model, noisy_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        if len(noisy_open_ids) > 1 and args.noisy_open_train:
            noisy_open_dataset = CustomDataSet(img_train[noisy_open_ids], txt_train[noisy_open_ids], label_train_noisy[noisy_open_ids], label_train_ori[noisy_open_ids])
            noisy_loader = torch.utils.data.DataLoader(noisy_open_dataset, batch_size=args.batch_size, shuffle=True)
            train_noisy_open(model, noisy_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        # 验证
        model.eval()
        t_imgs_fea, t_imgs_pred, t_txts_fea, t_txts_pred, t_labels = [], [], [], [], []
        with torch.no_grad():
            for imgs, txts, labels_noisy, labels_ori, index in valid_loader:
                if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels_ori.cuda()
                t_view1_feature, t_view2_feature = model(imgs, txts)
                t_view1_predict = F.softmax(t_view1_feature.view([t_view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_view2_predict = F.softmax(t_view2_feature.view([t_view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_imgs_fea.append(t_view1_feature.cpu().numpy())
                t_imgs_pred.append(t_view1_predict.cpu().numpy())
                t_txts_fea.append(t_view2_feature.cpu().numpy())
                t_txts_pred.append(t_view2_predict.cpu().numpy())
                t_labels.append(labels.cpu().numpy())
        t_imgs_fea = np.concatenate(t_imgs_fea)
        t_imgs_pred = np.concatenate(t_imgs_pred)
        t_txts_fea = np.concatenate(t_txts_fea)
        t_txts_pred = np.concatenate(t_txts_pred)
                    # t_labels = np.concatenate(t_labels).argmax(1)
        t_labels = np.concatenate(t_labels)
        img2txt = fx_calc_map_multilabel(t_imgs_fea, t_txts_fea, t_labels, metric='cosine')
        txt2img = fx_calc_map_multilabel(t_txts_fea, t_imgs_fea, t_labels, metric='cosine')
        MAPI2T_list.append(img2txt)
        MAPT2I_list.append(txt2img)
        num_val = t_labels.shape[0]
        img_acc = np.sum(np.argmax(t_imgs_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val
        txt_acc = np.sum(np.argmax(t_txts_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val

        print('Img2Txt: %.4f  Txt2Img: %.4f Imgacc: %.4f  Txtacc: %.4f lr: %g'%(img2txt, txt2img, img_acc, txt_acc, optimizer.param_groups[0]['lr']))
        if (img2txt + txt2img) / 2 > best_acc:
            best_acc = (img2txt + txt2img) / 2
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list, Clean_num_all_list
def warm_up_epoch(model, train_loader, valid_loader, optimizer, barycenters, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        tmp1 = - (labels * view1_predict.log()).sum(1)
        tmp2 = - (labels * view2_predict.log()).sum(1)
        term1 = (tmp1 + tmp2).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion(view1_feature, view2_feature, args)
        loss = args.lamda * term1 + args.alpha * term2
        loss.backward()
        optimizer.step()

    model.eval()
    t_imgs_fea, t_imgs_pred, t_txts_fea, t_txts_pred, t_labels = [], [], [], [], []
    with torch.no_grad():
        for imgs, txts, labels_noisy, labels_ori, index in valid_loader:
            if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    txts = txts.cuda()
                    labels = labels_ori.cuda()
            t_view1_feature, t_view2_feature = model(imgs, txts)
            t_view1_predict = F.softmax(t_view1_feature.view([t_view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
            t_view2_predict = F.softmax(t_view2_feature.view([t_view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
            t_imgs_fea.append(t_view1_feature.cpu().numpy())
            t_imgs_pred.append(t_view1_predict.cpu().numpy())
            t_txts_fea.append(t_view2_feature.cpu().numpy())
            t_txts_pred.append(t_view2_predict.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
    t_imgs_fea = np.concatenate(t_imgs_fea)
    t_imgs_pred = np.concatenate(t_imgs_pred)
    t_txts_fea = np.concatenate(t_txts_fea)
    t_txts_pred = np.concatenate(t_txts_pred)
                # t_labels = np.concatenate(t_labels).argmax(1)
    t_labels = np.concatenate(t_labels)
    img2txt = fx_calc_map_multilabel(t_imgs_fea, t_txts_fea, t_labels, metric='cosine')
    txt2img = fx_calc_map_multilabel(t_txts_fea, t_imgs_fea, t_labels, metric='cosine')
    num_val = t_labels.shape[0]
    img_acc = np.sum(np.argmax(t_imgs_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val
    txt_acc = np.sum(np.argmax(t_txts_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val

    print('Loss: %.4f Img2Txt: %.4f  Txt2Img: %.4f Imgacc: %.4f  Txtacc: %.4f lr: %g'%(loss, img2txt, txt2img, img_acc, txt_acc, optimizer.param_groups[0]['lr']))
def train_pure(model, train_loader, optimizer, barycenters, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        tmp1 = - (labels * view1_predict.log()).sum(1)
        tmp2 = - (labels * view2_predict.log()).sum(1)
        term1 = (tmp1 + tmp2).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion(view1_feature, view2_feature, args)
        loss = args.lamda * term1 + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_hard(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        # ---- 基于梯度大小的权重 ----
        # grad1 = (img_soft_labels_reordered[index] * labels).sum(1).detach()  # (N,)
        # grad2 = (txt_soft_labels_reordered[index] * labels).sum(1).detach()  # (N,)
        grad1 = (view1_predict * labels).sum(1).detach()  # (N,)
        grad2 = (view2_predict * labels).sum(1).detach()  # (N,)
        weights = torch.stack([grad1, grad2], dim=0)  # [2, batch, class]
        weights = torch.softmax(weights, dim=0)
        weight = (weights[0] * grad1 + weights[1] * grad2).unsqueeze(1)
        if args.hard_weight:
            tmp1 = - (weight * labels * view1_predict.log()).sum(1)
            tmp2 = - (weight * labels * view2_predict.log()).sum(1)
            term1 = (tmp1 + tmp2).mean()
        else:
            tmp1 = - (labels * view1_predict.log()).sum(1)
            tmp2 = - (labels * view2_predict.log()).sum(1)
            term1 = (tmp1 + tmp2).mean()
        term2 = cross_modal_contrastive_criterion(view1_feature, view2_feature, args)
        loss = args.lamda * term1 + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_noisy_close(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    all_pseudo_indices, all_ori_labels = [], []
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        num_classes = view1_predict.shape[1]
        if args.noisy_close_lamda:
            # view1_pesdo_labels = args.lamda_lc * labels + (1 - args.lamda_lc) * img_soft_labels_reordered[index]
            # view2_pesdo_labels = args.lamda_lc * labels + (1 - args.lamda_lc) * txt_soft_labels_reordered[index]
            view1_pesdo_labels = args.lamda_lc * labels + (1 - args.lamda_lc) * view1_predict
            view2_pesdo_labels = args.lamda_lc * labels + (1 - args.lamda_lc) * view2_predict
            tmp = torch.stack([view1_pesdo_labels, view2_pesdo_labels], dim=0)  # [2, batch, class]
            tmp = torch.softmax(tmp, dim=0)
            pesdo_labels = (tmp[0] * view1_pesdo_labels + tmp[1] * view2_pesdo_labels).detach()
            pesdo_labels_idx = torch.argmax(pesdo_labels, dim=1).detach()
        else:
            # soft_labels = 1 - (1- img_soft_labels_reordered[index]) * (1 - txt_soft_labels_reordered[index])
            soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
            pesdo_labels_idx = torch.argmax(soft_labels, dim=1).detach()
            pesdo_labels = F.one_hot(pesdo_labels_idx, num_classes=num_classes).float().detach()
        # soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        # pesdo_labels_idx = torch.argmax(soft_labels, dim=1).detach()
        # pesdo_labels = F.one_hot(pesdo_labels_idx, num_classes=num_classes).float()
        # 收集伪标签和原始标签
        all_pseudo_indices.extend(pesdo_labels_idx.cpu().numpy())
        all_ori_labels.extend(torch.argmax(labels_ori, dim=1).cpu().numpy())

        tmp1 = (pesdo_labels - view1_predict).abs().sum(1)
        tmp2 = (pesdo_labels - view2_predict).abs().sum(1)
        term1 = (tmp1 + tmp2).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion(view1_feature, view2_feature, args)
        loss = args.lamda * term1 + args.alpha * term2
        loss.backward()
        optimizer.step()
    # ✅ 计算整个 epoch 的翻新正确率（伪标签 vs 原始噪声标签）
    pseudo_array = np.array(all_pseudo_indices)
    true_array = np.array(all_ori_labels)
    acc = accuracy_score(true_array, pseudo_array)

    print(f"Correct acc: {acc:.4f}")
    
def train_noisy_open(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    all_pseudo_indices, all_ori_labels = [], []
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion(view1_feature, view2_feature, args)
        term3 = open_set_loss(view1_predict, view2_predict)
        loss = args.alpha * term2 + args.beta * term3
        loss.backward()
        optimizer.step()


def divide_sample(model, train_loader, barycenters, input_data_par, args):
    model.eval()
    t_imgs_fea, t_txts_fea, t_labels, t_labels_ori, sample_ids = [], [], [], [], []
    clean_indexs, noisy_indexs = [], []

    with torch.no_grad():
        for imgs, txts, labels_noisy, labels_ori, index in train_loader:
            clean_index = torch.argmax(labels_noisy, dim=1) == torch.argmax(labels_ori, dim=1)
            noisy_index = ~clean_index
            clean_indexs.append(index[clean_index])
            noisy_indexs.append(index[noisy_index])

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels_noisy = labels_noisy.cuda()

            t_view1_feature, t_view2_feature = model(imgs, txts)
            t_imgs_fea.append(t_view1_feature.cpu())
            t_txts_fea.append(t_view2_feature.cpu())
            t_labels.append(labels_noisy.cpu())
            t_labels_ori.append(labels_ori.cpu())
            sample_ids.append(index)

    t_imgs_fea = torch.cat(t_imgs_fea, dim=0)
    t_txts_fea = torch.cat(t_txts_fea, dim=0)    
    t_labels = torch.cat(t_labels, dim=0)
    t_labels_ori = torch.cat(t_labels_ori, dim=0)
    sample_ids = torch.cat(sample_ids, dim=0)
    clean_indexs = torch.cat(clean_indexs, dim=0)
    noisy_indexs = torch.cat(noisy_indexs, dim=0)
    barycenter_class = [i for i in range(args.closeset_num)]
    if args.uniform_weight:
        barycenter_weight = np.ones(args.closeset_num)/(args.closeset_num)
    else:
        N, C = t_labels.shape
        barycenter_weight = np.sum(t_labels.numpy(), axis=0) / N
    t_all_fea = torch.cat([t_imgs_fea, t_txts_fea], dim=0).numpy()
    t_labels_all = torch.cat([t_labels, t_labels], dim=0).numpy()
    train_sample_weight = np.ones(t_all_fea.shape[0])/t_all_fea.shape[0]
    cost_matrix = cdist(barycenters, t_all_fea, metric='cosine') 
       
    opt_result = label_reg(barycenter_weight, train_sample_weight, cost_matrix, barycenter_class, args.closeset_num, args.lambda_ot)
    prob, label_propagation = label_propagation_analysis(opt_result, barycenter_class, args.closeset_num)
    mean = np.mean(prob, axis=0, keepdims=True)
    std = np.std(prob, axis=0, keepdims=True) + 1e-12
    normalized_prob = (prob - mean) / std
    exp_prob = np.exp(normalized_prob / args.tau)
    softmax_prob = exp_prob / np.sum(exp_prob, axis=0, keepdims=True)
    
    num_img = t_imgs_fea.shape[0]
    num_txt = t_txts_fea.shape[0]
    
    label_prop_img = label_propagation[:num_img]
    label_prop_txt = label_propagation[num_img:num_img+num_txt]
    
    softmax_img = softmax_prob[:, :num_img]
    softmax_txt = softmax_prob[:, num_img:num_img+num_txt]
    
    true_labels = torch.argmax(t_labels, dim=1)   # shape=(2N,)
    true_labels_ori = torch.argmax(t_labels_ori, dim=1) 
    true_labels_img = true_labels
    true_labels_txt = true_labels

    # 预测标签 (label_propagation 已经是 numpy，可以转 torch)
    pred_img = torch.tensor(label_prop_img)  # shape=(N,)
    pred_txt = torch.tensor(label_prop_txt)  # shape=(N,)

    # 图像/文本预测是否正确
    img_clean = (pred_img == true_labels_img)
    txt_clean = (pred_txt == true_labels_txt)

    # 基础三类
    pure_clean_mask = img_clean & txt_clean       # 干净集
    hard_mask = img_clean ^ txt_clean             # Hard集
    noisy_mask = ~(img_clean | txt_clean)         # 噪声集

    # 在噪声集上细分 (必集噪声 vs 开集噪声)
    noisy_closeset_mask = noisy_mask & (pred_img == pred_txt)   # 必集噪声
    noisy_openset_mask = noisy_mask & (pred_img != pred_txt)   # 开集噪声

    # 用 sample_ids 取对应 ID
    pure_clean_ids = sample_ids[:num_img][pure_clean_mask]
    hard_ids = sample_ids[:num_img][hard_mask]
    noisy_close_ids = sample_ids[:num_img][noisy_closeset_mask]
    noisy_open_ids = sample_ids[:num_img][noisy_openset_mask]
    
    # 真实干净样本 ID (t_labels == t_labels_ori)
    real_clean_mask = (true_labels == true_labels_ori)
    real_clean_ids = sample_ids[real_clean_mask]

    # 真实必集噪声 ID (标签在闭集范围内，但是 != 原始标签)
    true_labels = torch.argmax(t_labels_ori, dim=1)  # (N,)
    real_noisy_close_mask = (true_labels < input_data_par['closeset_num']) & (~real_clean_mask)
    real_noisy_close_ids = sample_ids[real_noisy_close_mask]

    # 真实开集噪声 ID (标签在开集范围)
    real_noisy_open_mask = (true_labels >= input_data_par['closeset_num']) & \
                        (true_labels < input_data_par['closeset_num'] + input_data_par['openset_num'])
    real_noisy_open_ids = sample_ids[real_noisy_open_mask]

    # --- 准确率计算函数 ---
    def compute_selection_accuracy(pred_ids, real_ids):
        pred_set = set(pred_ids.cpu().numpy().tolist())
        real_set = set(real_ids.cpu().numpy().tolist())
        correct = len(pred_set & real_set)
        accuracy = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        return accuracy, correct, len(real_set)

    # --- 计算三类准确率 ---
    clean_acc, clean_correct, clean_total = compute_selection_accuracy(pure_clean_ids, real_clean_ids)
    hard_acc, hard_correct, hard_total = compute_selection_accuracy(hard_ids, real_clean_ids)
    noisy_close_acc, close_correct, close_total = compute_selection_accuracy(noisy_close_ids, real_noisy_close_ids)
    noisy_open_acc, open_correct, open_total = compute_selection_accuracy(noisy_open_ids, real_noisy_open_ids)

    print(f"Pure-clean: Acc={clean_acc:.4f} ({clean_correct}/{pure_clean_ids.shape[0]})")
    print(f"Hard: Acc={hard_acc:.4f} ({hard_correct}/{hard_ids.shape[0]})")
    print(f"Noisy-close: Acc={noisy_close_acc:.4f} ({close_correct}/{noisy_close_ids.shape[0]})")
    print(f"Noisy-open: Acc={noisy_open_acc:.4f} ({open_correct}/{noisy_open_ids.shape[0]})")
    return pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, softmax_img.T, softmax_txt.T, clean_correct, pure_clean_ids.shape[0], clean_total


def get_barycenters(model, train_loader, input_data_par, args):
    # 计算每个类别的特征质心
    model.eval()
    train_feature_view1, train_feature_view2 = [], []
    train_label = np.array([]).astype('int16')

    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            txts = txts.cuda()
            labels = labels_noisy.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        train_feature_view1.append(view1_feature.cpu().detach().numpy())
        train_feature_view2.append(view2_feature.cpu().detach().numpy())
        train_label = np.concatenate((train_label, np.argmax(labels.cpu().detach().numpy(), axis=1)))
    train_feature_view1 = np.concatenate(train_feature_view1, axis=0)
    train_feature_view2 = np.concatenate(train_feature_view2, axis=0)
    

    barycenters = []
    for class_id in range(input_data_par['closeset_num']):
        sample = np.where(train_label==class_id)
        sample_feature_view1 = train_feature_view1[sample]
        sample_feature_view2 = train_feature_view2[sample]
        sample_feature = np.concatenate((sample_feature_view1, sample_feature_view2), axis=0)
        center = k_barycenter(sample_feature.transpose(), args.barycenter_number, args.lambda_ot)
        barycenters += center.transpose().tolist()
    barycenters = np.array(barycenters)
    return barycenters.astype('float32')
