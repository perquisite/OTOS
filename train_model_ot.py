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
from scipy.io import loadmat
from utils import *
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
from load_data import CustomDataSet
from opt import k_barycenter, label_reg, label_propagation_analysis


def train_model(model, input_data_par, optimizer, args):
    img_train, txt_train, label_train_ori, label_train_noisy = input_data_par['img_train'], input_data_par['text_train'], input_data_par['label_train_ori'], input_data_par['label_train_noisy']
    img_valid, txt_valid, label_valid = input_data_par['img_valid'], input_data_par['text_valid'], input_data_par['label_valid']
    args.closeset_num = input_data_par['closeset_num']
    train_clean_indices = np.argmax(label_train_noisy, axis=1) == np.argmax(label_train_ori, axis=1)
    train_clean_indices = np.where(train_clean_indices)[0]
    num_epochs = args.MAX_EPOCH
    test_img_acc_history = []
    test_txt_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list ,Clean_num_all_list = [], [], [], [], []

    train_dataset = CustomDataSet(img_train, txt_train, label_train_noisy, label_train_ori)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = CustomDataSet(img_valid, txt_valid, label_valid, label_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    since = time.time()
    for epoch in range(args.warm_up_epoch):
        print('Warm up epoch {}/{}'.format(epoch+1, args.warm_up_epoch))
        print('-' * 20)
        # get barycenters
        if args.random_barycenters:
            barycenters = np.random.rand(args.closeset_num, args.output_dim).astype('float32')
        elif args.mean_barycenters:
            barycenters = get_barycenters_mean(model, train_loader, input_data_par, args)
        else:
            barycenters = get_barycenters(model, train_loader, input_data_par, args)
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        warm_up_epoch(model, train_loader, valid_loader, optimizer, barycenters, args)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        # get barycenters
        barycenters = get_barycenters(model, train_loader, input_data_par, args)
        
        if args.GMM:
            data = divide_sample_gmm(model, train_loader, barycenters, input_data_par, epoch, args)
        elif args.sim:
            data = divide_sample_sim(model, train_loader, barycenters, input_data_par, epoch, args)
        else:
            data = divide_sample(model, train_loader, barycenters, input_data_par, epoch, args)
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        (pure_clean_ids, hard_ids, noisy_close_ids, noisy_open_ids, clean_correct, clean_selected, clean_total) = data
        Clean_num_selected_list.append(clean_correct)
        num_selected_list.append(clean_selected)
        Clean_num_all_list.append(clean_total)
        if len(pure_clean_ids) > 1:
            pure_dataset = CustomDataSet(img_train[pure_clean_ids], txt_train[pure_clean_ids], label_train_noisy[pure_clean_ids], label_train_ori[pure_clean_ids])
            pure_loader = torch.utils.data.DataLoader(pure_dataset, batch_size=args.batch_size, shuffle=True)
            train_pure(model, pure_loader, optimizer, barycenters, args)
        if len(hard_ids) > 1 and args.hard_train:
            print(len(hard_ids))
            hard_dataset = CustomDataSet(img_train[hard_ids], txt_train[hard_ids], label_train_noisy[hard_ids], label_train_ori[hard_ids])
            hard_loader = torch.utils.data.DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=True)
            train_hard(model, hard_loader, optimizer, barycenters, args)
        if len(noisy_close_ids) + len(noisy_open_ids) > 1:
            if args.noisy_distinguish:
                print('distinguish open and closed set noise')
                if len(noisy_close_ids) > 1 and args.noisy_close_train:
                    noisy_close_dataset = CustomDataSet(img_train[noisy_close_ids], txt_train[noisy_close_ids], label_train_noisy[noisy_close_ids], label_train_ori[noisy_close_ids])
                    noisy_loader = torch.utils.data.DataLoader(noisy_close_dataset, batch_size=args.batch_size, shuffle=True)
                    train_noisy_close(model, noisy_loader, optimizer, barycenters, args)
                if len(noisy_open_ids) > 1 and args.noisy_open_train:
                    noisy_open_dataset = CustomDataSet(img_train[noisy_open_ids], txt_train[noisy_open_ids], label_train_noisy[noisy_open_ids], label_train_ori[noisy_open_ids])
                    noisy_loader = torch.utils.data.DataLoader(noisy_open_dataset, batch_size=args.batch_size, shuffle=True)
                    train_noisy_open(model, noisy_loader, optimizer, barycenters, args)
            else:
                print('not distinguish open and closed set noise')
                noisy_ids = torch.cat([noisy_close_ids, noisy_open_ids], dim=0)
                noisy_close_dataset = CustomDataSet(img_train[noisy_ids], txt_train[noisy_ids], label_train_noisy[noisy_ids], label_train_ori[noisy_ids])
                noisy_loader = torch.utils.data.DataLoader(noisy_close_dataset, batch_size=args.batch_size, shuffle=True)
                train_noisy_close(model, noisy_loader, optimizer, barycenters, args)
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
            best_barycenters = barycenters.detach().cpu().numpy()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list, Clean_num_all_list, best_barycenters

    
