# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_SSD300.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_SSD300.aocx

# Run inference tool
# GTSDB
model=deploy_gtsdb_ssd_vgg16_reduced_300_510-0210.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/mxnet
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/object_detection_sample_ssd -i $OPENVINO_ROOTDIR/test_img/GTSDB -m $model -d HETERO:FPGA,CPU

# Classification
# model=deploy_resnet-20_mxnet_GT_32by32_3-0025.xml
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/mxnet
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/classification_sample -i $HOME/Desktop/GTSRB/Final_Training/Images/00000/00000_00008.ppm -m $model -d HETERO:FPGA,CPU