# TrafficSignBench
A benchmark for deep learning frameworks on traffic sign recognition task

## Usage:
* Download the code and change working directory to the folder.
```bash
git clone https://github.com/owensgroup/TrafficSignBench.git
cd TrafficSignBench
```
* Prepare Dataset. Currently only the GTSRB dataset is supported.
```bash
export DATASET_ROOT=/path/of/your/choice
./prepare_dataset.sh
```

* Run the benchmark. Change the setting in the script file if necessary.
```bash
./bench_script.sh
```

## Notice:
Neon throws a "Floating point exception" when running with CUDA 9. You could fix it by replacing line 844 of the kernel_specs.py file under /path/to/your/python/site-packages/nervananeon-2.6.0-py3.5.egg/neon/backends/ with:
```python
run_command([ "/usr/local/cuda-8.0/bin/ptxas -v -arch", arch, "-o", cubin_file, ptx_file, ";" ] + maxas_i + [sass_file, cubin_file])
```

## Bugs to be fixed
* MXNet and PyTorch generate "out of memory" bugs when they are launched after any other frameworks.
