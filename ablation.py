import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nuswide')  # xmedia, wiki, nuswide, INRIA-Websearch
parser.add_argument("--GPU", type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.dataset
    logging = datasets
    output_dim_list = [512]
    noisy_ratio_list = [0.2,0.4,0.8]
    alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2]
    beta_os_list = [7]
    batch_size_list = [256]
    threshold_list = [0.5]
    gpu = args.GPU
    if datasets == 'wiki':
        alpha_list = [1]
        beta_os_list = [0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
        threshold_list = [0.3,0.4,0.5]
        seed_list = [10]
        logging = 'wiki_ablation_v2'
    elif datasets == 'xmedia':
        alpha_list = [0.9]
        beta_os_list = [7]
        seed_list = [10]
        logging = 'xmedia_ablation_0.2_0.8'
    elif datasets == 'INRIA-Websearch':
        alpha_list = [0.3]
        beta_os_list = [7]
        seed_list = [10]
        logging = 'INRIA-Websearch_ablation_0.2_0.8'
    for seed in seed_list:
        for output_dim in output_dim_list:
            for batch_size in batch_size_list:
                for threshold in threshold_list:
                    for beta_os in beta_os_list:
                        for alpha in alpha_list:
                            for noisy_ratio in noisy_ratio_list:
                                # 正常结果       
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉constractive learning alpha = 0
                                # result = subprocess.run(
                                # ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(0), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                #  '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                # print(result.stdout)
                                # if result.stderr:
                                #     print(result.stderr)
                                # 去掉openset loss beta = 0
                                # result = subprocess.run(
                                # ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(0), '--batch_size', str(batch_size),
                                #  '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                # print(result.stdout)
                                # if result.stderr:
                                #     print(result.stderr)
                                # 去掉openset loss beta_os = 0
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(0), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉label loss lamda = 0
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--lamda', str(0), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉nagetive pairs weight lambda_cl = 1
                                # result = subprocess.run(
                                # ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                #  '--lamda_lc', str(lamda_lc), '--lambda_cl', str(1), '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', '1'], capture_output=True, text=True)
                                # print(result.stdout)
                                # if result.stderr:
                                #     print(result.stderr)
                                # 去掉hard weight hard_weight = False
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--hard_weight', 'False', '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉noisy_close noisy_close_train = False
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--noisy_close_train', 'False', '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉noisy_open noisy_open_train = False
                                result = subprocess.run(
                                ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta_os), '--batch_size', str(batch_size),
                                 '--threshold', str(threshold), '--noisy_open_train', 'False', '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                print(result.stdout)
                                if result.stderr:
                                    print(result.stderr)
                                # 去掉warm up warm_up_epoch = 0
                                # result = subprocess.run(
                                # ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta), '--batch_size', str(batch_size),
                                #  '--lamda_lc', str(lamda_lc), '--warm_up_epoch', '0', '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', str(gpu)], capture_output=True, text=True)
                                # print(result.stdout)
                                # if result.stderr:
                                #     print(result.stderr)
                                # 去掉uniform_weight uniform_weight = True
                                # result = subprocess.run(
                                # ['python', 'main.py', '--seed', str(seed), '--dataset', datasets, '--alpha', str(alpha), '--beta_os', str(beta), '--batch_size', str(batch_size),
                                #  '--lamda_lc', str(lamda_lc), '--uniform_weight', 'True', '--noisy_ratio', str(noisy_ratio), '--output_dim', str(output_dim), '--logging', logging, '--GPU', '0'], capture_output=True, text=True)
                                # print(result.stdout)
                                # if result.stderr:
                                #     print(result.stderr)
