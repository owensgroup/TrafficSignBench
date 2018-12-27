# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
# R3
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_VGG.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_VGG.aocx
# R4
# FP11
# aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx
# export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx
# FP16
# # VGG
# aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
# export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
# # Non-VGG
# aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
# export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx

# Run inference tool
# GTSDB
model=deploy_gtsdb_ssd_vgg16_reduced_300_510-0102.xml
# model=deploy_gtsdb_ssd_mobilenet_v1_300_510-0102.xml
# model=deploy_gtsdb_ssd_mobilenet_v2_300_510-0102.xml
# model=deploy_gtsdb_ssdlite_mobilenet_v2_300_510-0204.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/mxnet
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/object_detection_sample_ssd -i $OPENVINO_ROOTDIR/test_img/GTSDB -m $model -d HETERO:FPGA,CPU

# Classification
# model=deploy_resnet-20_mxnet_GT_32by32_3-0025.xml
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/mxnet
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/classification_sample -i $HOME/Desktop/GTSRB/Final_Training/Images/00000/00000_00008.ppm -m $model -d HETERO:FPGA,CPU