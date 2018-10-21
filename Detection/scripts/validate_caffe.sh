# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP11_SSD300.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP11_SSD300.aocx

# # SSD512
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/ssd/caffe/ssd_vgg_512
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/VOCdevkit/VOC2007/Annotations -i $HOME/Documents/data/VOCdevkit -m ssd512.xml -ODc ./VOC_SSD_Classes.txt -ODsubdir JPEGImages -d CPU

# GTSDB
# model_name=VGG
# iter=30000
# model=${model_name}_SSD_510x300_100_40_Square_${iter}.xml

model_name=MobileNet
iter=80000
model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.xml

# model_name=MobileNetV2
# iter=50000
# model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.xml

# model_name=MobileNetV2
# iter=200000
# model=${model_name}_SSDLite_510x300_100_40_Square_1_${iter}.xml

cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/caffe/${model_name}
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m $model -ODc $HOME/Documents/data/GTSDB_SSD_Classes_caffe.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU


# HETERO:FPGA,CPU