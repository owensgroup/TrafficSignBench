# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
aocl program acl0 $OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP11_SSD300.aocx
export DLA_AOCX=$OPENVINO_ROOTDIR/a10_devkit_bitstreams/2-0-1_A10DK_FP11_SSD300.aocx

# GTSDB
# model=deploy_gtsdb_ssd_vgg16_reduced_300_510-0210.xml
# model=deploy_gtsdb_ssd_mobilenet_v1_300_510-0102.xml
# model=deploy_gtsdb_ssd_mobilenet_v2_300_510-0102.xml
model=deploy_gtsdb_ssdlite_mobilenet_v2_300_510-0204.xml
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/mxnet
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m $model -ODc $HOME/Documents/data/GTSDB_SSD_Classes_mxnet.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU

# GTSRB
# model=deploy_mxnet_GT_48by48_3-0025.xml
# size=32
# model=deploy_resnet-20_mxnet_GT_${size}by${size}_3-0025.xml
# size=64
# model=deploy_resnet-32_mxnet_GT_${size}by${size}_3-0025.xml
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/classification/mxnet
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t C -i $HOME/Desktop/GTSRB/Final_Test/pp_imgs_clahe_${size}_${size}/labels.txt -m $model -d HETERO:FPGA,CPU