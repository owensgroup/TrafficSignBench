# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_ResNet.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_ResNet.aocx

# ResNet-20
model=idsia_48by48.xml
# model=resnet-20_32by32.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/tf
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/classification_sample -i $HOME/Desktop/GTSRB/Final_Training/Images/00000/00000_00008.ppm -m $model -d HETERO:FPGA,CPU