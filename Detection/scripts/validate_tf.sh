# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_ResNet.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_ResNet.aocx

# model=idsia_48by48.xml
# size=32
# model=resnet-20_${size}by${size}.xml
size=64
model=resnet-32_${size}by${size}.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/tf
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t C -i $HOME/Desktop/GTSRB/Final_Test/pp_imgs_clahe_${size}_${size}/labels.txt -m $model -d HETERO:FPGA,CPU