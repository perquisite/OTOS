
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
import matplotlib.pyplot as plt
from load_data import CustomDataSet
from opt import k_barycenter, label_reg, label_propagation_analysis

from sklearn.mixture import GaussianMixture

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

def cross_modal_contrastive_criterion_supervised(img_fea, txt_fea, labels, pred=None, subset_name='', args=None):
    device = img_fea.device
    labels = labels.float()
    N, C = labels.shape

    # ---------- sim logits ----------
    logits_i2t = (img_fea @ txt_fea.t())        # (N, N)   corresponds to Psi(z_i^v, z_j^t)
    logits_t2i = (txt_fea @ img_fea.t())        # (N, N)   corresponds to Psi(z_i^t, z_j^v)

    diag_idx = torch.arange(N, device=device)

    # ---------- 构造同类掩码 P(i) (不含自身) ----------
    same_label = (labels @ labels.t()) > 0     # (N, N) boolean
    same_label.fill_diagonal_(False)           # remove self

    # ---------- 计算 log denom (stable) ----------
    # log_denom_i = log sum_j exp(logits_i2t[i,j])
    log_denom_i2t = torch.logsumexp(logits_i2t, dim=1)  # (N,)
    log_denom_t2i = torch.logsumexp(logits_t2i, dim=1)  # (N,)

    # ---------- 主正对项 (i -> t) ----------
    diag_logits_i2t = logits_i2t[diag_idx, diag_idx]    # (N,)
    main_pos_i2t = - (diag_logits_i2t - log_denom_i2t).mean()

    # ---------- 主正对项 (t -> i) ----------
    diag_logits_t2i = logits_t2i[diag_idx, diag_idx]    # (N,)
    main_pos_t2i = - (diag_logits_t2i - log_denom_t2i).mean()

    # ---------- 总损失 ----------
    loss = args.alpha * (main_pos_i2t + main_pos_t2i)
    return loss


def open_set_loss(soft_labels):
    max_conf = soft_labels.max(dim=1)[0]
    loss = -(1 - max_conf).log().mean()
    return loss


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim

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
        # tmp1 = - (labels * view1_predict.log()).sum(1)
        # tmp2 = - (labels * view2_predict.log()).sum(1)
        # term1 = (tmp1 + tmp2).mean()
        
        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        term1 = -(labels * soft_labels.log()).sum(1).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion_supervised(view1_feature, view2_feature, labels, subset_name='warm_up', args=args)
        loss = args.lamda * term1 + term2
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
        # tmp1 = - (labels * view1_predict.log()).sum(1)
        # tmp2 = - (labels * view2_predict.log()).sum(1)
        # term1 = (tmp1 + tmp2).mean()
        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        term1 = -(labels * soft_labels.log()).sum(1).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion_supervised(view1_feature, view2_feature, labels, subset_name='pure', args=args)
        loss = args.lamda * term1 + term2
        loss.backward()
        optimizer.step()
def train_hard(model, train_loader, optimizer, barycenters, args):
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
        
        grad1 = (view1_predict * labels).sum(1).detach()  # (N,)
        grad2 = (view2_predict * labels).sum(1).detach()  # (N,)
        weights = torch.stack([grad1, grad2], dim=0)  # [2, batch, class]
        weights = torch.softmax(weights, dim=0)
        weight = (weights[0] * grad1 + weights[1] * grad2).unsqueeze(1)
        if args.hard_weight:
            # tmp1 = - (weight * labels * view1_predict.log()).sum(1)
            # tmp2 = - (weight * labels * view2_predict.log()).sum(1)
            # term1 = (tmp1 + tmp2).mean()
            soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
            term1 = -(weight * labels * soft_labels.log()).sum(1).mean()
        else:
            tmp1 = - (labels * view1_predict.log()).sum(1)
            tmp2 = - (labels * view2_predict.log()).sum(1)
            term1 = (tmp1 + tmp2).mean()
        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        term2 = cross_modal_contrastive_criterion_supervised(view1_feature, view2_feature, labels, pred=soft_labels, subset_name='hard', args=args)
        loss = args.lamda * term1 + term2
        loss.backward()
        optimizer.step()
def train_noisy_close(model, train_loader, optimizer, barycenters, args):
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
        
        # tmp1 = (1 - pesdo_labels * view1_predict).sum(1)
        # tmp2 = (1 - pesdo_labels * view2_predict).sum(1)
        # term1 = (tmp1 + tmp2).mean()
        if args.noisy_close_weight:
            grad1 = (view1_predict * pesdo_labels).sum(1).detach()  # (N,)
            grad2 = (view2_predict * pesdo_labels).sum(1).detach()  # (N,)
            weights = torch.stack([grad1, grad2], dim=0) 
            weights = torch.softmax(weights, dim=0)
            weight = (weights[0] * grad1 + weights[1] * grad2).unsqueeze(1)
            term1 = -(weight * labels * soft_labels.log()).sum(1).mean()
        else:
            term1 = (1 - pesdo_labels * soft_labels).sum(1).mean()
        # loss = term1.mean()
        term2 = cross_modal_contrastive_criterion_supervised(view1_feature, view2_feature, pesdo_labels, pred=soft_labels, subset_name='noisy_closed', args=args)
        loss = args.lamda * term1 + term2
        loss.backward()
        optimizer.step()
    # ✅ 计算整个 epoch 的翻新正确率（伪标签 vs 原始噪声标签）
    pseudo_array = np.array(all_pseudo_indices)
    true_array = np.array(all_ori_labels)
    acc = accuracy_score(true_array, pseudo_array)

    print(f"Correct acc: {acc:.4f}")
    
def train_noisy_open(model, train_loader, optimizer, barycenters, args):
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
        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        term2 = cross_modal_contrastive_criterion_supervised(view1_feature, view2_feature, labels, subset_name='noisy_open', args=args)
        term3 = open_set_loss(soft_labels)
        
        loss = term2 + args.beta_os * term3
        loss.backward()
        optimizer.step()


def divide_sample(model, train_loader, barycenters, input_data_par, epoch, args):
    model.eval()
    barycenters = torch.tensor(barycenters, requires_grad = False)
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
    
    num_img = t_imgs_fea.shape[0]
    num_txt = t_txts_fea.shape[0]
    
    label_prop_img = label_propagation[:num_img]
    label_prop_txt = label_propagation[num_img:num_img+num_txt]
    
    
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
    
    view1_predict = F.softmax(t_imgs_fea.view([t_imgs_fea.shape[0], -1]).mm(barycenters.T), dim=1)
    view2_predict = F.softmax(t_txts_fea.view([t_txts_fea.shape[0], -1]).mm(barycenters.T), dim=1)
    soft_labels = 1 - (1 - view1_predict) * (1 - view2_predict)
    # 取置信度（每一行最大值）
    confidence, _ = torch.max(soft_labels, dim=1)
    max_value, min_value = max(confidence), min(confidence)
    confidence_norm = (confidence-min_value)/(max_value-min_value)

    # --- 新的开/闭集噪声划分 ---

    if args.adaptive_threshold:
        non_noisy_mask = pure_clean_mask | hard_mask
        non_noisy_confidence = confidence_norm[non_noisy_mask]
        args.threshold = np.median(non_noisy_confidence)
    # 只在 noisy_mask 中划分
    noisy_close_mask = noisy_mask & (confidence_norm >= args.threshold)   # 高置信度 = 闭集噪声
    noisy_open_mask  = noisy_mask & (confidence_norm < args.threshold)    # 低置信度 = 开集噪声

    # 用 sample_ids 取对应 ID
    pure_clean_ids = sample_ids[:num_img][pure_clean_mask]
    hard_ids = sample_ids[:num_img][hard_mask]
    noisy_close_ids = sample_ids[:num_img][noisy_close_mask]
    noisy_open_ids = sample_ids[:num_img][noisy_open_mask]
    
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
    # barycenters = update_barycenters(t_imgs_fea[pure_clean_ids], t_txts_fea[pure_clean_ids], t_labels[pure_clean_ids], barycenters, input_data_par, args)
    if epoch in range(4, 100, 5):
        plot_prob(args.dataset, t_imgs_fea, t_txts_fea, barycenters, real_noisy_close_ids, real_noisy_open_ids, sample_ids, epoch)
    return pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, clean_correct, pure_clean_ids.shape[0], clean_total


def divide_sample_gmm(model, train_loader, barycenters, input_data_par, epoch, args):
    model.eval()
    t_imgs_fea, t_txts_fea, t_labels, t_labels_ori, sample_ids = [], [], [], [], []
    clean_indexs, noisy_indexs = [], []
    barycenters = torch.tensor(barycenters, requires_grad = False)

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

    t_imgs_fea = torch.cat(t_imgs_fea, dim=0)      # (N, D)
    t_txts_fea = torch.cat(t_txts_fea, dim=0)      # (N, D)
    t_labels = torch.cat(t_labels, dim=0)          # (N, C)
    t_labels_ori = torch.cat(t_labels_ori, dim=0)  # (N, C)
    sample_ids = torch.cat(sample_ids, dim=0)
    N = t_imgs_fea.shape[0]

    # 转为 numpy barycenters (C, D)
    barycenters_np = barycenters # shape: (num_classes, D)
    num_classes = barycenters_np.shape[0]

    # 获取每个样本的带噪标签（整数）
    noisy_labels_int = torch.argmax(t_labels, dim=1).cpu().numpy()  # (N,)

    # === Step 1: 计算每个样本与其对应类别质心的余弦相似度 ===
    def compute_cosine_sim_with_barycenter(features, label_indices, barycenters):
        # features: (N, D), label_indices: (N,), barycenters: (C, D)
        sims = []
        for i in range(features.shape[0]):
            c = label_indices[i]
            if c >= num_classes:
                # 超出闭集范围，跳过或设为最小相似度（但此处假设 noisy label ∈ [0, C-1]）
                sim = -1.0
            else:
                feat = features[i].numpy()
                center = barycenters[c]
                # cosine similarity
                sim = np.dot(feat, center) / (np.linalg.norm(feat) * np.linalg.norm(center) + 1e-8)
            sims.append(sim)
        return np.array(sims)  # (N,)

    sim_img = compute_cosine_sim_with_barycenter(t_imgs_fea, noisy_labels_int, barycenters_np)  # (N,)
    sim_txt = compute_cosine_sim_with_barycenter(t_txts_fea, noisy_labels_int, barycenters_np)  # (N,)

    # === Step 2: 用双成分 GMM 建模相似度分布（干净 vs 噪声）===
    # 注意：我们分别对图像和文本建模？还是联合？这里按模态分开更合理
    def fit_two_component_gmm(sim_scores):
        # reshape to (N, 1) for GMM
        X = sim_scores.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='full',
            random_state=0,
            max_iter=100
        )
        gmm.fit(X)

        # Identify which component is "high" (clean): the one with higher mean
        means = gmm.means_.flatten()
        high_comp = int(np.argmax(means))  # index of high-similarity component

        # Compute posterior prob of belonging to high component
        responsibilities = gmm.predict_proba(X)  # (N, 2)
        prob_high = responsibilities[:, high_comp]  # (N,)

        return prob_high, gmm, high_comp

    prob_high_img, gmm_img, _ = fit_two_component_gmm(sim_img)
    prob_high_txt, gmm_txt, _ = fit_two_component_gmm(sim_txt)

    # Convert to torch
    prob_high_img = torch.from_numpy(prob_high_img)
    prob_high_txt = torch.from_numpy(prob_high_txt)

    # === Step 3: 判定每个模态是否干净（高相似度分量后验 > 0.5）===
    img_clean = (prob_high_img > 0.5)
    txt_clean = (prob_high_txt > 0.5)

    # 基础三类划分
    pure_clean_mask = img_clean & txt_clean
    hard_mask = img_clean ^ txt_clean
    noisy_mask = ~(img_clean | txt_clean)

    # === Step 4: 构造置信度用于开/闭集划分 ===
    # 使用 prob_high 作为置信度（已在 [0,1]）
    confidence_img = prob_high_img
    confidence_txt = prob_high_txt
    # 融合图文置信度：类似之前逻辑
    confidence = 1 - (1 - confidence_img) * (1 - confidence_txt)
    confidence_norm = confidence  # already in [0,1]

    # 开/闭集噪声划分（仅在 noisy_mask 中）
    noisy_close_mask = noisy_mask & (confidence_norm >= args.threshold)
    noisy_open_mask = noisy_mask & (confidence_norm < args.threshold)

    # ID 提取
    pure_clean_ids = sample_ids[pure_clean_mask]
    hard_ids = sample_ids[hard_mask]
    noisy_close_ids = sample_ids[noisy_close_mask]
    noisy_open_ids = sample_ids[noisy_open_mask]

    # --- 真实标签用于评估 ---
    true_labels_noisy = torch.argmax(t_labels, dim=1)
    true_labels_ori = torch.argmax(t_labels_ori, dim=1)

    real_clean_mask = (true_labels_noisy == true_labels_ori)
    real_clean_ids = sample_ids[real_clean_mask]

    true_gt = true_labels_ori
    real_noisy_close_mask = (true_gt < input_data_par['closeset_num']) & (~real_clean_mask)
    real_noisy_close_ids = sample_ids[real_noisy_close_mask]

    real_noisy_open_mask = (true_gt >= input_data_par['closeset_num']) & \
                           (true_gt < input_data_par['closeset_num'] + input_data_par['openset_num'])
    real_noisy_open_ids = sample_ids[real_noisy_open_mask]

    # --- 准确率计算 ---
    def compute_selection_accuracy(pred_ids, real_ids):
        pred_set = set(pred_ids.cpu().numpy().tolist())
        real_set = set(real_ids.cpu().numpy().tolist())
        correct = len(pred_set & real_set)
        accuracy = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        return accuracy, correct, len(real_set)

    clean_acc, clean_correct, clean_total = compute_selection_accuracy(pure_clean_ids, real_clean_ids)
    hard_acc, hard_correct, hard_total = compute_selection_accuracy(hard_ids, real_clean_ids)
    noisy_close_acc, close_correct, close_total = compute_selection_accuracy(noisy_close_ids, real_noisy_close_ids)
    noisy_open_acc, open_correct, open_total = compute_selection_accuracy(noisy_open_ids, real_noisy_open_ids)

    print(f"Pure-clean: Acc={clean_acc:.4f} ({clean_correct}/{pure_clean_ids.shape[0]})")
    print(f"Hard: Acc={hard_acc:.4f} ({hard_correct}/{hard_ids.shape[0]})")
    print(f"Noisy-close: Acc={noisy_close_acc:.4f} ({close_correct}/{noisy_close_ids.shape[0]})")
    print(f"Noisy-open: Acc={noisy_open_acc:.4f} ({open_correct}/{noisy_open_ids.shape[0]})")

    # 可选：绘图（传入原始 barycenters）
    if epoch in range(4, 100, 5):
        plot_prob(args.dataset, t_imgs_fea, t_txts_fea, barycenters,
                  real_noisy_close_ids, real_noisy_open_ids, sample_ids, epoch)

    return pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, clean_correct, pure_clean_ids.shape[0], clean_total

def divide_sample_sim(model, train_loader, barycenters, input_data_par, epoch, args):
    model.eval()
    t_imgs_fea, t_txts_fea, t_labels, t_labels_ori, sample_ids = [], [], [], [], []
    clean_indexs, noisy_indexs = [], []
    barycenters = torch.tensor(barycenters, requires_grad = False)

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

    t_imgs_fea = torch.cat(t_imgs_fea, dim=0)      # (N, D)
    t_txts_fea = torch.cat(t_txts_fea, dim=0)      # (N, D)
    t_labels = torch.cat(t_labels, dim=0)          # (N, C)
    t_labels_ori = torch.cat(t_labels_ori, dim=0)  # (N, C)
    sample_ids = torch.cat(sample_ids, dim=0)
    N = t_imgs_fea.shape[0]

    # barycenters: (C, D)
    barycenters = barycenters.float()  # ensure float
    num_classes = barycenters.shape[0]

    # Normalize features and barycenters for cosine similarity
    def normalize(x):
        return F.normalize(x, p=2, dim=1)

    barycenters_norm = normalize(barycenters).cpu()  # (C, D)
    imgs_fea_norm = normalize(t_imgs_fea)      # (N, D)
    txts_fea_norm = normalize(t_txts_fea)      # (N, D)

    # Cosine similarity = dot product after L2 normalization
    sim_img = torch.mm(imgs_fea_norm, barycenters_norm.t())  # (N, C)
    sim_txt = torch.mm(txts_fea_norm, barycenters_norm.t())  # (N, C)

    # Pseudo labels: argmax of similarity
    pseudo_label_img = torch.argmax(sim_img, dim=1)  # (N,)
    pseudo_label_txt = torch.argmax(sim_txt, dim=1)  # (N,)

    # Noisy labels (from input)
    noisy_label = torch.argmax(t_labels, dim=1)  # (N,)

    # === 判定每个模态是否干净：pseudo == noisy_label ===
    img_clean = (pseudo_label_img == noisy_label)  # (N,)
    txt_clean = (pseudo_label_txt == noisy_label)  # (N,)

    # 基础三类划分
    pure_clean_mask = img_clean & txt_clean
    hard_mask = img_clean ^ txt_clean
    noisy_mask = ~(img_clean | txt_clean)

    # === 置信度：取最大相似度值（已在 [0,1] 因为 normalized）===
    confidence_img, _ = torch.max(sim_img, dim=1)  # (N,)
    confidence_txt, _ = torch.max(sim_txt, dim=1)  # (N,)
    # 融合图文置信度
    confidence = 1 - (1 - confidence_img) * (1 - confidence_txt)
    confidence_norm = confidence  # already in [0, 1]

    # === 自适应阈值：pure + hard 样本的平均置信度 ===
    non_noisy_mask = pure_clean_mask | hard_mask
    if non_noisy_mask.sum() > 0:
        adaptive_threshold = confidence_norm[non_noisy_mask].mean().item()
    else:
        adaptive_threshold = max(confidence_norm.mean().item(), 0.1)

    # --- 开/闭集噪声划分（仅在 noisy_mask 中）---
    noisy_close_mask = noisy_mask & (confidence_norm >= adaptive_threshold)
    noisy_open_mask = noisy_mask & (confidence_norm < adaptive_threshold)

    # ID 提取
    pure_clean_ids = sample_ids[pure_clean_mask]
    hard_ids = sample_ids[hard_mask]
    noisy_close_ids = sample_ids[noisy_close_mask]
    noisy_open_ids = sample_ids[noisy_open_mask]

    # --- 真实标签用于评估 ---
    true_labels_noisy = torch.argmax(t_labels, dim=1)
    true_labels_ori = torch.argmax(t_labels_ori, dim=1)

    real_clean_mask = (true_labels_noisy == true_labels_ori)
    real_clean_ids = sample_ids[real_clean_mask]

    true_gt = true_labels_ori
    real_noisy_close_mask = (true_gt < input_data_par['closeset_num']) & (~real_clean_mask)
    real_noisy_close_ids = sample_ids[real_noisy_close_mask]

    real_noisy_open_mask = (true_gt >= input_data_par['closeset_num']) & \
                           (true_gt < input_data_par['closeset_num'] + input_data_par['openset_num'])
    real_noisy_open_ids = sample_ids[real_noisy_open_mask]

    # --- 准确率计算 ---
    def compute_selection_accuracy(pred_ids, real_ids):
        pred_set = set(pred_ids.cpu().numpy().tolist())
        real_set = set(real_ids.cpu().numpy().tolist())
        correct = len(pred_set & real_set)
        accuracy = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        return accuracy, correct, len(real_set)

    clean_acc, clean_correct, clean_total = compute_selection_accuracy(pure_clean_ids, real_clean_ids)
    hard_acc, hard_correct, hard_total = compute_selection_accuracy(hard_ids, real_clean_ids)
    noisy_close_acc, close_correct, close_total = compute_selection_accuracy(noisy_close_ids, real_noisy_close_ids)
    noisy_open_acc, open_correct, open_total = compute_selection_accuracy(noisy_open_ids, real_noisy_open_ids)

    print(f"Pure-clean: Acc={clean_acc:.4f} ({clean_correct}/{pure_clean_ids.shape[0]})")
    print(f"Hard: Acc={hard_acc:.4f} ({hard_correct}/{hard_ids.shape[0]})")
    print(f"Noisy-close: Acc={noisy_close_acc:.4f} ({close_correct}/{noisy_close_ids.shape[0]})")
    print(f"Noisy-open: Acc={noisy_open_acc:.4f} ({open_correct}/{noisy_open_ids.shape[0]})")

    # 可选：绘图
    if epoch in range(4, 100, 5):
        plot_prob(args.dataset, t_imgs_fea, t_txts_fea, barycenters,
                  real_noisy_close_ids, real_noisy_open_ids, sample_ids, epoch)

    return pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, clean_correct, pure_clean_ids.shape[0], clean_total


def plot_prob(dataset, view1_feature, view2_feature, barycenters, real_noisy_close_ids, real_noisy_open_ids, sample_ids, epoch):
    id2idx = {sid.item(): idx for idx, sid in enumerate(sample_ids)}
    noisy_close_indices = torch.tensor([id2idx[sid.item()] for sid in real_noisy_close_ids]).cuda()
    noisy_open_indices = torch.tensor([id2idx[sid.item()] for sid in real_noisy_open_ids]).cuda()
    
    view1_feature, view2_feature, barycenters = torch.tensor(view1_feature).cuda(), torch.tensor(view2_feature).cuda(), torch.tensor(barycenters).float().cuda()
    view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
    view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
    soft_labels = 1 - (1 - view1_predict) * (1 - view2_predict)

    openset_soft_labels, closedset_soft_labels = torch.max(soft_labels[noisy_open_indices], dim=1).values, torch.max(soft_labels[noisy_close_indices], dim=1).values
    max_value = max(max(openset_soft_labels), max(closedset_soft_labels))
    min_value = min(min(openset_soft_labels), min(closedset_soft_labels))
    openset_soft_labels, closedset_soft_labels = (openset_soft_labels - min_value)/(max_value - min_value), (closedset_soft_labels - min_value)/(max_value - min_value)

    plt.figure(figsize=(12, 10))

    # 设置坐标轴边框和刻度
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tick_params(axis='both', width=3)

    # 绘制直方图
    plt.hist(openset_soft_labels.cpu().numpy(), bins=50, density=True, histtype='stepfilled', 
             color='red', alpha=0.4, label='Open-set')
    plt.hist(closedset_soft_labels.cpu().numpy(), bins=50, density=True, histtype='stepfilled', 
             color='green', alpha=0.4, label='Closed-set')

    # 图例
    # if dataset == 'xmedia':
    #     plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=35)
    plt.legend(loc='upper center', fontsize=35, edgecolor='black', bbox_to_anchor=(0.5, 1.03))

    # 坐标标签和刻度
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=46)
    plt.xlabel('Normalized Affinity', fontsize=50)
    plt.ylabel('Density', fontsize=50)
    plt.rc('axes', linewidth=1.5)

    plt.savefig('img/' + dataset + '_sim_hist_' + str(epoch) + '.pdf', bbox_inches='tight')



    
    
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
def get_barycenters_mean(model, train_loader, input_data_par, args):
    # 计算每个类别的特征质心（使用所有样本的平均特征）
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
    
    train_feature_view1 = np.concatenate(train_feature_view1, axis=0)  # (N, D)
    train_feature_view2 = np.concatenate(train_feature_view2, axis=0)  # (N, D)

    barycenters = []
    for class_id in range(input_data_par['closeset_num']):
        # 找到属于当前类的所有样本索引（基于 noisy label）
        sample_indices = np.where(train_label == class_id)[0]
        
        if len(sample_indices) == 0:
            # 若该类无样本，用零向量或全局均值（这里用零向量，后续可替换）
            feature_dim = train_feature_view1.shape[1]
            center = np.zeros(feature_dim)
        else:
            # 提取该类所有图像和文本特征
            class_feat_view1 = train_feature_view1[sample_indices]  # (M, D)
            class_feat_view2 = train_feature_view2[sample_indices]  # (M, D)
            # 合并图文特征：(2M, D)
            all_class_features = np.concatenate([class_feat_view1, class_feat_view2], axis=0)
            # 计算均值作为质心
            center = np.mean(all_class_features, axis=0)  # (D,)
        
        barycenters.append(center)
    
    barycenters = np.array(barycenters)  # (closeset_num, D)
    return barycenters.astype('float32')
def update_barycenters(train_feature_view1, train_feature_view2, labels, barycenters, input_data_par, args):
    for class_id in range(input_data_par['closeset_num']):
        sample = np.where(labels==class_id)
        sample_feature_view1 = train_feature_view1[sample]
        sample_feature_view2 = train_feature_view2[sample]
        sample_feature = np.concatenate((sample_feature_view1, sample_feature_view2), axis=0)
        if sample_feature.shape[0]>0:
            center = k_barycenter(sample_feature.transpose(), args.barycenter_number, args.lambda_ot)
            barycenters += center.transpose().tolist()
            barycenters[class_id] = center
    barycenters = np.array(barycenters)
    return barycenters.astype('float32')

