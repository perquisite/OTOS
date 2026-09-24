from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
import scipy.io as sio
from torch.utils.data import DataLoader
import numpy as np
import h5py
import random
import os

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels_noisy,
            labels_ori):
        self.images = images
        self.texts = texts
        self.labels_noisy = labels_noisy
        self.labels_ori = labels_ori
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label_noisy = self.labels_noisy[index]
        labels_ori = self.labels_ori[index]
        return img, text, label_noisy, labels_ori, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels_noisy)
        return count

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

def get_noisylabels_closeset(labels, noisy_ratio, noise_mode):
    labels_num = np.sum(labels, axis=1)
    class_num = labels.shape[1]
    inx = np.arange(class_num)
    np.random.shuffle(inx)
    transition = {i: i for i in range(class_num)}
    half_num = int(class_num // 2)
    for i in range(half_num):
        transition[inx[i]] = int(inx[half_num + i])
    data_num = labels.shape[0]
    idx = list(range(data_num))
    random.shuffle(idx)
    num_noise = int(noisy_ratio * data_num)
    noise_idx = idx[:num_noise]
    noise_label = np.zeros((data_num, class_num), dtype=int)
    for i in range(data_num):
        if i in noise_idx:
            if noise_mode == 'sym':
                tmp = int(labels_num[i])
                index = np.random.choice(class_num, tmp, replace=False)
                noise_label[i, index] = 1
            elif noise_mode == 'asym':
                pass
        else:
            noise_label[i, :] = labels[i, :]
    return noise_label

def get_noisylabels_openset(labels, noisy_ratio, openset_ratio):
    data_num = labels.shape[0]
    class_num = labels.shape[1]
    closeset_num = int(class_num * (1 - openset_ratio))
    openset_classes = list(range(int(class_num * (1 - openset_ratio)), class_num))
    closeset_classes = list(range(0, int(class_num * (1 - openset_ratio))))

    labels_num = np.sum(labels, axis=1)
    gt_labels = np.argmax(labels, axis=1)

    noisy_labels = np.copy(labels)

    for i in range(data_num):
        true_label = gt_labels[i]
        # Open-set 类别样本
        if true_label in openset_classes:
            # 替换为 closeset 中的随机标签
            new_label = random.choice(closeset_classes)
            noisy_labels[i, :] = 0
            noisy_labels[i, new_label] = 1

        # Close-set 类别样本
        else:
            if random.random() < noisy_ratio:  # noisy sample
                # 替换为另一个 closeset 中的随机标签
                new_label = random.choice(closeset_classes)
                noisy_labels[i, :] = 0
                noisy_labels[i, new_label] = 1
            else:
                pass  # 保持原标签不变

    return noisy_labels[:, 0:closeset_num]
def get_loader(data_name, args):
    batch_size = args.batch_size
    noisy_ratio = args.noisy_ratio
    openset_ratio = args.openset_ratio
    noise_mode = args.noise_mode
    np.random.seed(1)
    if data_name == 'wiki':
        valid_len = 231
        path = '/home/puruitao/datasets/wiki.mat'
        data = sio.loadmat(path)
        img_train = data['train_imgs_deep']
        text_train = data['train_texts_doc']
        label_train_img = data['train_imgs_labels'].reshape([-1,1]).astype('int16') 

        img_test = data['test_imgs_deep']
        text_test = data['test_texts_doc']
        label_test_img = data['test_imgs_labels'].reshape([-1,1]).astype('int16') 

        img_valid = img_test[0:valid_len]
        text_valid = text_test[0:valid_len]
        label_valid_img = label_test_img[0:valid_len]

        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test_img = label_test_img[valid_len:]
    elif data_name == 'xmedia':
        valid_len = 500
        path = '/home/puruitao/datasets/XMediaFeatures.mat'
        all_data = sio.loadmat(path)
        img_test = all_data['I_te_CNN'].astype('float32')  # Features of test set for image data, CNN feature
        img_train = all_data['I_tr_CNN'].astype('float32')   # Features of training set for image data, CNN feature
        text_test = all_data['T_te_BOW'].astype('float32')  # Features of test set for text data, BOW feature
        text_train = all_data['T_tr_BOW'].astype('float32')   # Features of training set for text data, BOW feature

        label_test_img = all_data['teImgCat'].reshape([-1,1]).astype('int64')-1 # category label of test set for image data
        label_train_img = all_data['trImgCat'].reshape([-1,1]).astype('int64')-1 # category label of training set for image data

        img_valid = img_test[0:valid_len]
        text_valid = text_test[0:valid_len]
        label_valid_img = label_test_img[0:valid_len]

        img_test = img_test[valid_len:]
        text_test =  text_test[valid_len:]
        label_test_img = label_test_img[valid_len:]
    elif data_name == 'INRIA-Websearch':
        path = '/home/puruitao/datasets/INRIA-Websearch.mat'
        data = sio.loadmat(path)
        img_train = data['tr_img'].astype('float32')
        text_train = data['tr_txt'].astype('float32')
        label_train_img = data['tr_img_lab'].reshape([-1,1]).astype('int16')

        img_valid = data['val_img'].astype('float32')
        text_valid = data['val_txt'].astype('float32')
        label_valid_img = data['val_img_lab'].reshape([-1,1]).astype('int16')

        img_test = data['te_img'].astype('float32')
        text_test = data['te_txt'].astype('float32')
        label_test_img = data['te_img_lab'].reshape([-1,1]).astype('int16') 
    elif data_name == 'nuswide':
        valid_len = 0
        path = '/home/puruitao/datasets/nus_wide_deep_doc2vec-corr-ae.h5py'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
            img_train = file['train_imgs_deep'][:]
            text_train = file['train_texts'][:]
            label_train_img = file['train_imgs_labels'][:]

            img_valid = file['valid_imgs_deep'][:]
            text_valid = file['valid_texts'][:]
            label_valid_img = file['valid_imgs_labels'][:]

            img_test = file['test_imgs_deep'][:]
            text_test = file['test_texts'][:]
            label_test_img = file['test_imgs_labels'][:]

    elif data_name == 'xmedianet':
        valid_len = 4000
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/CrossNL/DSCMR/newData/XMediaNet5View_Doc2Vec.mat'
        all_data = sio.loadmat(path)
        all_train_data = all_data['train'][0]
        all_train_labels = all_data['train_labels'][0]
        all_valid_data = all_data['valid'][0]
        all_valid_labels = all_data['valid_labels'][0]
        all_test_data = all_data['test'][0]
        all_test_labels = all_data['test_labels'][0]

        img_train = all_train_data[0]
        text_train = all_train_data[1]
        label_train_img = all_train_labels[0].reshape([-1,1]).astype('int64') 

        img_valid = all_valid_data[0]
        text_valid = all_valid_data[1]
        label_valid_img = all_valid_labels[0].reshape([-1,1]).astype('int64') 

        img_test = all_test_data[0]
        text_test = all_test_data[1]
        label_test_img = all_test_labels[0].reshape([-1,1]).astype('int64')
     

        
    img_train = img_train.astype('float32')
    img_valid = img_valid.astype('float32')
    img_test = img_test.astype('float32')
    text_train = text_train.astype('float32')
    text_valid = text_valid.astype('float32')
    text_test = text_test.astype('float32')
    label_train = label_train_img
    label_valid = label_valid_img
    label_test = label_test_img
    if len(label_train.shape) == 1 or label_train.shape[1] == 1:
        label_train = ind2vec(label_train.reshape([-1,1])).astype('int16') 
        label_valid = ind2vec(label_valid.reshape([-1,1])).astype('int16') 
        label_test = ind2vec(label_test.reshape([-1,1])).astype('int16') 
    print('train shape: ', img_train.shape[0], 'valid shape:', img_valid.shape[0], 'test shape:', img_test.shape[0])
    root_dir = '/home/puruitao/datasets/noisy_labels_openset'
    noise_file = os.path.join(root_dir, data_name + '_noise_labels_%g_' %noisy_ratio) + noise_mode + '.mat'
    if os.path.exists(noise_file):
        label_noisy = sio.loadmat(noise_file)['noisy_label']
    elif noise_mode == 'openset':    #inject noise
        label_noisy = get_noisylabels_openset(label_train, noisy_ratio, openset_ratio)
        sio.savemat(noise_file,{'noisy_label':label_noisy})
    elif noise_mode in ['sym', 'asym']:
        label_noisy = get_noisylabels_closeset(label_train, noisy_ratio, noise_mode)
        sio.savemat(noise_file,{'noisy_label':label_noisy})
    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_train = img_train.shape[0]
    num_class = label_train.shape[1]
    closeset_num = int(num_class * (1 - openset_ratio))


    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_valid'] = img_valid
    input_data_par['text_valid'] = text_valid
    input_data_par['label_valid'] = label_valid
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['num_train'] = num_train
    input_data_par['label_train_ori'] = label_train
    input_data_par['label_train_noisy'] = label_noisy
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    input_data_par['closeset_num'] = closeset_num
    input_data_par['openset_num'] = num_class - closeset_num
    return input_data_par