from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import torch
import os
import gc

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from perturbed_layer.expand_layer import ExpandLayer
from utils import create_vnnlib_str, get_valid_data
       
def _process_expand_single_spec(args, model, benchmark_dir, w, x, y, logit, count, seed):
    try:
        base_name = f"{count}_{seed}_{args.model}_{w}"
        onnx_name = os.path.join('onnx', f"{base_name}.onnx")
        
        x_lb = torch.tensor([[0.0]])
        x_ub = torch.tensor([[w]])  
        specs = create_vnnlib_str(
            data_lb=x_lb, 
            data_ub=x_ub, 
            prediction=logit,
        )
        
        perturbed_layer = ExpandLayer(
            image=x.squeeze(0), 
            max_displacement=0.3
        ).cpu()
        perturbed_net = torch.nn.Sequential(perturbed_layer, model).cpu()
        perturbed_net.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                perturbed_net,
                x_lb,
                os.path.join(benchmark_dir, onnx_name),
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                },
                verbose=False
            )
        
        del x_lb, x_ub, perturbed_layer, perturbed_net
        gc.collect()
        
        res = []
        for i, spec in enumerate(specs):
            spec_name = os.path.join('vnnlib', f"{base_name}_{i}.vnnlib")
            with open(os.path.join(benchmark_dir, spec_name), 'w') as f:
                print(spec, file=f)
            res.append(f'{onnx_name},{spec_name},{args.timeout}')
        
        return {'success': True, 'result': res, 'task': (args, model, benchmark_dir, w, x, y, logit, count, seed)}
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Failed to process {count}th spec")
        return {'success': False, 'task': (args, model, benchmark_dir, w, x, y, logit, count, seed)}
            
             
def generate_expand_prop(args, model, test_loader, label_to_index, device):

    print(f'\n{"="*80}')
    print(f"Starting {args.model} expand specs generation...")
    print(f'{"="*80}\n')
    
    valid_data = get_valid_data(args, model, test_loader, label_to_index, device)

    benchmark_dir = os.path.join(args.benchmark_dir, f'expand_{args.model}')
    os.makedirs(os.path.join(benchmark_dir, 'vnnlib'), exist_ok=True)
    os.makedirs(os.path.join(benchmark_dir, 'onnx'), exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    tasks = []
    count = 0
    
    for w in [0.2, 0.5, 0.8]:
        for x, y, logit in valid_data:
            tasks.append((args, model, benchmark_dir, w, x, y, logit, count, args.seed))
            count += 1
    
    results = Parallel(n_jobs=4)(
        delayed(_process_expand_single_spec)(*task) for task in tqdm(tasks)
    )
    successful_results = []
    failed_results = []
    for r in results:
        if r['success']:
            successful_results.extend(r['result'])
        else:
            failed_results.append(r['task'])
    
    for failed_result in tqdm(failed_results):
        retry_result = _process_expand_single_spec(*failed_result['task'])
        if retry_result['success']:
            successful_results.extend(retry_result['result'])
    
    with open(os.path.join(benchmark_dir, f'instances.csv'), 'w') as f, open(os.path.join(benchmark_dir, f'command.sh'), 'w') as f2:
        for result in successful_results:
            print(result, file=f)
            print(f'python3 main.py --net {os.path.abspath(benchmark_dir)}/{result.split(",")[0]} --spec {os.path.abspath(benchmark_dir)}/{result.split(",")[1]} --timeout {args.timeout}', file=f2)
    
    print(f"[!] Expand: {len(successful_results)} specs")
    gc.collect()
    
    