# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_SSD300.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP16_SSD300.aocx

# Run inference tool
# GoogleNet
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/googlenet
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/classification_sample -i $OPENVINO_ROOTDIR/test_img/cat.jpg -m bvlc_googlenet.xml -d HETERO:FPGA,CPU

# SSD512
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/ssd/caffe/ssd_vgg_512
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/object_detection_sample_ssd -i $OPENVINO_ROOTDIR/test_img/fish-bike.png -m ssd512.xml -d HETERO:FPGA,CPU


# VGG-/MobileNet-SSD
# model=VGG_SSD_510x300_100_40_Square_30000.xml
model=MobileNet_SSD_510x300_100_40_Square_1_30000.xml
# model=MobileNetV2_SSD_510x300_100_40_Square_1_50000.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/caffe/40
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/object_detection_sample_ssd -i $OPENVINO_ROOTDIR/test_img/GTSDB -m $model -d HETERO:FPGA,CPU # -t 0.15