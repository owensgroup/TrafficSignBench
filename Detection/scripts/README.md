## OpenVINO scripts
This folder contains bash scripts for using OpenVINO, e.g. converting a pretrained model with the model optimizer, inferencing a image on an FPGA, and benchmarking inference speed and accuracy on a dataset.

These scripts are tested with Intel's [OpenVINO release R4 & R5](https://software.intel.com/en-us/articles/OpenVINO-RelNotes). Makes sure to run
```bash
export OPENVINO_ROOTDIR=/path/of/your/choice
```
before running the scripts.

### Notice
For Tensorflow we still use the model optimizer R3 for model conversion. Run the following commands to get the optimizer:
```bash
cd /path/of/your/choice
git clone https://github.com/opencv/dldt/tree/e607ee70212797cf9ca51dac5b7ac79f66a1c73f
cd dldt
cp -r model-optimizer $OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3
```
