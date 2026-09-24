import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuswide')  # xmedia, wiki, nuswide, INRIA-Websearch
parser.add_argument("--GPU", type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.dataset
    logging = datasets
    gpu = args.GPU
    output_dim = 512
    noisy_ratio = 0.6
    alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    beta_os_list = [0,1,2,3,4,5,6,7,8,9,10]
    threshold = 0.5
    batch_size = 256
    seed = 10
    beta_os_ori = 7
    if datasets == 'wiki':
        logging = 'wiki_para'
        alpha_ori = 0.3
    elif datasets == 'xmedia':
        logging = 'xmedia_para'
        alpha_ori = 0.9
    elif datasets == 'INRIA-Websearch':
        logging = 'INRIA-Websearch_para'
        alpha_ori = 0.3
    for alpha in alpha_list:
        logging_cur = logging + '_alpha'
        result = subprocess.run(
        ['python', 'main.py', '--seed', str(seed),  '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os_ori), '--batch_size', str(batch_size),
        '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging_cur, '--GPU', str(gpu)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    for beta_os in beta_os_list:
        logging_cur = logging + '_beta'
        result = subprocess.run(
        ['python', 'main.py', '--seed', str(seed),  '--dataset', datasets, '--alpha', str(alpha_ori), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
        '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging_cur, '--GPU', str(gpu)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
