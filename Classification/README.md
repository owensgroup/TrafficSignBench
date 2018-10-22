## Classification
Benchmarking popular deep learning frameworks, including CNTK, MXNet, Neon, Keras, PyTorch, and Tensorflow with models including
- IDSIA (32x32) (http://people.idsia.ch/~ciresan/data/ijcnn2011.pdf)
- ResNet-20 (32x32, 48x48, 64x64)
- ResNet-32 (32x32, 48x48, 64x64) (https://arxiv.org/pdf/1512.03385.pdf)  
on a traffic sign dataset GTSRB (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Usage:
* Download the code and change working directory to the folder.
```bash
git clone https://github.com/owensgroup/TrafficSignBench.git
cd TrafficSignBench
```
* Prepare Dataset. Currently only the GTSRB dataset is supported.
```bash
export DATA=/path/of/your/choice
./prepare_dataset.sh
```

* Run the benchmark. Change the setting in the script file if necessary.
```bash
./bench_script.sh
```

## Notice:
Neon throws a "Floating point exception" when running with CUDA 9. You could fix it by replacing line 844 of the kernel_specs.py file under 
```
/path/to/your/python/site-packages/nervananeon-2.6.0-py3.5.egg/neon/backends/
# e.g. ~/miniconda3/envs/neon/lib/python3.5/site-packages/nervananeon-2.6.0-py3.5.egg/neon/backends/
```
with:
```python
run_command([ "/usr/local/cuda-8.0/bin/ptxas -v -arch", arch, "-o", cubin_file, ptx_file, ";" ] + maxas_i + [sass_file, cubin_file])
```
Ref: https://github.com/xingjinglu/PerfAILibs/blob/master/README.md

## Bugs to be fixed
* MXNet and PyTorch generate "out of memory" bugs when they are launched after any other frameworks.
