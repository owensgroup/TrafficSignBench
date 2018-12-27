# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
############################# R4
# FP11
# aocx="4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx"
# FP16
# aocx="4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx"

############################## R5
# FP11
# aocx="5-0_A10DK_FP11_MobileNet_Clamp.aocx"
# FP16
# MobileNet
aocx="5-0_A10DK_FP16_MobileNet_Clamp.aocx"
# # ResNet
# aocx="5-0_A10DK_FP16_ResNet_TinyYolo.aocx"

aocl program acl0 "${OPENVINO_ROOTDIR}/bitstreams/a10_devkit_bitstreams/${aocx}"
export DLA_AOCX="$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/${aocx}"

# Run inference tool
# GTSDB
# model=MobileNet/frozen_inference_graph.xml
model=MobileNetV2/frozen_inference_graph.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/tf
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/object_detection_sample_ssd -i $OPENVINO_ROOTDIR/test_img/GTSDB -m $model -d HETERO:FPGA,CPU

# GTSRB
# model=idsia_48by48.xml
# model=resnet-20_32by32.xml
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/tf
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/classification_sample -i $HOME/Desktop/GTSRB/Final_Training/Images/00000/00000_00008.ppm -m $model -d HETERO:FPGA,CPU
