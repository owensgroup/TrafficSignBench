fp=$1
model_name=$2
if [ -z "$fp" ]
then
	fp="FP11"
fi
if [ -z "$model_name" ]
then
	model_name="VGG"
fi
printf $"Args: ${fp}, ${model_name}\n"

# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
if [ "$fp" == "FP11" ]; then
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx
# FP16
elif [ "$model_name" == "VGG" ]; then # VGG
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
else # Non-VGG
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
fi

# GTSDB
if [ "$model_name" == "VGG" ]; then
	model=deploy_gtsdb_ssd_vgg16_reduced_300_510-0102.xml
elif [ "$model_name" == "MobileNet" ]; then
	model=deploy_gtsdb_ssd_mobilenet_v1_300_510-0102.xml
else # MobileNetV2
	model=deploy_gtsdb_ssdlite_mobilenet_v2_300_510-0204.xml
fi
# model=deploy_gtsdb_ssd_mobilenet_v2_300_510-0102.xml
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