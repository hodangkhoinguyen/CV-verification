# Formal Verification of Geometric Robustness for Deep Neural Networks

## Train network
```
python training.py --dataset mnist --model fnn2
python training.py --dataset mnist --model fnn4

python training.py --dataset fashionmnist --model fnn2
python training.py --dataset fashionmnist --model fnn4
```

# Generate properties
```
python generate_property.py --dataset mnist --benchmark_dir benchmark_mnist  --model fnn2
python generate_property.py --dataset mnist --benchmark_dir benchmark_mnist  --model fnn4

python generate_property.py --dataset fashionmnist --benchmark_dir benchmark_fashion  --model fnn2
python generate_property.py --dataset fashionmnist --benchmark_dir benchmark_fashion  --model fnn4
```

# Run all verification tasks
```
python run_verify.py --benchmark_dir benchmark_mnist/ --verifier_dir neuralsat/src/ --option all --result_dir result_mnist/
python run_verify.py --benchmark_dir benchmark_fashion/ --verifier_dir neuralsat/src/ --option all --result_dir result_fashion/
```