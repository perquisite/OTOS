import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def plot_prob(clean_weights, noisy_weights, title, epoch):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = plt.gca()

    ax.spines['top'].set_linewidth(2)   
    ax.spines['right'].set_linewidth(2)   
    ax.spines['bottom'].set_linewidth(2) 
    ax.spines['left'].set_linewidth(2)   
    ax.tick_params(axis='both', width=3)

    plt.rcParams['font.family'] = 'Times New Roman'

    # plt.title(title, fontsize=46, fontname='Times New Roman', fontweight='bold')
#     plt.xlabel('Per-sample loss', fontsize=42, fontname='Times New Roman')
#     plt.ylabel('Number of samples', fontsize=42, fontname='Times New Roman')
    ax.hist(noisy_weights, bins=50, density=True, histtype='stepfilled', color='red', alpha=0.4,
            label='False labeled instances')
    ax.hist(clean_weights, bins=50, density=True, histtype='stepfilled', color='green', alpha=0.4,
            label='True labeled instances')
    # ax.hist(uncertainties[clean_no_value_index], bins=50, density=False, histtype='stepfilled', color='blue', alpha=0.4,
            # label='Uninformative')
            #label='All')
    # ax.legend(loc='best', fontsize=38, edgecolor='black')
    ax.legend(loc='upper center', fontsize=45, edgecolor='black', bbox_to_anchor=(0.5, 1.05))
#     plt.xticks(np.arange(0,1.1,0.1), fontsize=46, fontname='Times New Roman')
#     plt.yticks([0, 100, 200], fontsize=46, fontname='Times New Roman')
    plt.xticks(fontsize=46, fontname='Times New Roman')
    plt.yticks(fontsize=46, fontname='Times New Roman')
    plt.xlabel('Weight', fontsize=50, fontname='Times New Roman')
    plt.ylabel('Density', fontsize=50, fontname='Times New Roman')
#     plt.xlim(-0.05, 1.05)
    #     plt.ylim(0, 190)

    
    plt.rc('axes', linewidth=1.5)
    plt.tight_layout()
    plt.savefig('SPL_weights_'+ str(epoch) + '.pdf')
dataset = 'INRIA-Websearch' # INRIA-Websearch
noisy_ratio = 0.6
tp = 5
all_data = loadmat('SPL_weights/' + dataset + '_weights_%.1f_%d_'%(noisy_ratio,tp) +'.mat')
clean_weights_list = all_data['clean_weights_list']
noisy_weights_list = all_data['noisy_weights_list']
MAX_EPOCH = all_data['max_epoch']
epochs = [5,10,100]
for epoch in epochs:
        plot_prob(clean_weights_list[epoch], noisy_weights_list[epoch], 'Epoch: ' + str(epoch), epoch)
