import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuswide')  # xmedia, wiki, nuswide, INRIA-Websearch

if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.dataset
    logging = datasets
    output_dim_list = [512]
    noisy_ratio_list = [0.2,0.4,0.6,0.8]
    alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2]
    beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    beta_os_list = [0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
    threshold_list = [0.5,0.6,0.4]
    batch_size_list = [256]
    if datasets == 'wiki':
        seed_list = [10]
        alpha_list = [1]
        beta_os_list = [7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5]
        threshold_list = [0.3,0.4,0.5]
        logging = 'wiki_findMAP_v2'
    elif datasets == 'xmedia':
        seed_list = [10]
        alpha_list = [1]
        beta_os_list = [7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5]
        threshold_list = [0.3,0.4,0.5]
        logging = 'xmedia_findMAP_v2'
    elif datasets == 'INRIA-Websearch':
        alpha_list = [1]
        beta_os_list = [7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5]
        threshold_list = [0.3,0.4,0.5]
        seed_list = [10]
        logging = 'INRIA-Websearch_findMAP_v2'
    elif datasets == 'nuswide':
        seed_list = [10]
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        beta_os_list = [7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5]
        logging = 'nuswide_findMAP'
    for seed in seed_list:
        for output_dim in output_dim_list:
            for batch_size in batch_size_list:
                for threshold in threshold_list:
                    for beta_os in beta_os_list:
                        for alpha in alpha_list:
                                # 正常结果
                            for noisy_ratio in noisy_ratio_list:
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed),  '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', '1'], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
