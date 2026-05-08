import tqdm
import os
import torch
import argparse
import time

def run_neuralsat(args, onnx_path, vnnlib_path, output_path, timeout):
    result_path = f'{output_path}.txt'
    log_path = f'{output_path}.log'
    if os.path.exists(result_path):
        status = open(result_path).read().strip().split(',')[0]
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
        if not (status == 'error' or status == "timeout"):
            return status

    os.chdir(args.verifier_dir)

    cmd = f'timeout {timeout}s'
    cmd += f' python3 -W ignore main.py --verbosity=2'
    cmd += f' --net {onnx_path} --spec {vnnlib_path} --timeout {timeout}'
    cmd += f' --result_file {result_path}'

    setting_path = os.path.join(args.home_dir, f'neuralsat_config.json')
    assert os.path.exists(setting_path), f"Setting file does not exist: {setting_path=}"
    print(setting_path)
    cmd += f' --setting_file {setting_path}'

    cmd += f' --export_runtime'
    cmd += f' > {log_path} 2>&1'
    print(cmd)
    tic = time.time()
    os.system(cmd)
    toc = time.time()
    
    if os.path.exists(result_path):
        status = open(result_path).read().strip().split(',')[0]
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
    else:
        status = 'error'
        with open(result_path, 'w') as f:
            print(f'{status},{toc - tic}', file=f)
    os.chdir(args.home_dir)
    return status

def get_path(*folder_list):
    res = "."
    for i in folder_list:
        res = os.path.join(res, i)
    return res

def get_benchmark_list(dir):
    folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    return folders

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", type=str, required=True, help="Root directory for benchmark")
    p.add_argument("--result_dir", type=str, required=True, help="Root directory for result")
    p.add_argument("--verifier_dir", type=str, required=True, help="Verifier directory")
    p.add_argument("--timeout", type=int, default=600, help="Timeout")
    p.add_argument("--device", type=str, default="cuda", help="Device to run the verifier")
    p.add_argument('--option', choices=['random', 'first', 'all'], type=str, required=True)
    args = p.parse_args()

    args.home_dir = os.getcwd()
    args.verifier_dir = os.path.abspath(os.path.join(args.home_dir, args.verifier_dir))
    
    return args

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args = parse_args()

    benchmark_dir = args.benchmark_dir
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    print(f'{benchmark_dir = }')
    print(f'{result_dir = }')

    benchmark_list = get_benchmark_list(benchmark_dir)


    count = 0
    for benchmark_name in benchmark_list:
        output_dir = get_path(result_dir, benchmark_name)
        os.makedirs(output_dir, exist_ok=True)

        benchmark_path = os.path.join(benchmark_dir, benchmark_name)
        instance_csv  = f'{benchmark_path}/instances.csv'
        result_csv_path = f'{output_dir}/results.csv'
        result_csv = open(result_csv_path, "w")

        instances = []
        for line in open(instance_csv):
            line = line.strip()
            onnx_path, vnnlib_path, _ = line.split(',')
            onnx_path = os.path.abspath(os.path.join(benchmark_path, onnx_path))
            vnnlib_path = os.path.abspath(os.path.join(benchmark_path, vnnlib_path))
            assert os.path.exists(onnx_path), f"ONNX file does not exist: {onnx_path=}"
            assert os.path.exists(vnnlib_path), f"VNNLIB file does not exist: {vnnlib_path=}"
            instances.append((onnx_path, vnnlib_path))

        if args.option == "random":
            ...
        elif args.option == "first":
            instances = instances[:10] # Adjust this line for the number of FIRST instances
        print(f"{benchmark_name=} {len(instances)=}")

        pbar = tqdm.tqdm(instances)
        pbar.set_description(f'Benchmark {benchmark_name} (timeout={args.timeout})')

        for idx, (onnx_path, vnnlib_path) in enumerate(pbar):
            onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
            vnnlib_name = os.path.splitext(os.path.basename(vnnlib_path))[0]            
            output_path = os.path.abspath(f'{output_dir}/net_{onnx_name}_spec_{vnnlib_name}')
            status = run_neuralsat(
                args=args,
                onnx_path=onnx_path,
                vnnlib_path=vnnlib_path,
                output_path=output_path,
                timeout=args.timeout
            )

            result_csv.write(f"{benchmark_name}\t{onnx_path}\t{vnnlib_path}\t{status}\n")
        result_csv.close()
