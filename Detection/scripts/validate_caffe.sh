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

# # SSD512
# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/ssd/caffe/ssd_vgg_512
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/benchmark_app -i $HOME/Documents/data/VOCdevkit/VOC2007/JPEGImages -m ssd512.xml -d CPU -api async

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
	iter=30000
	model=${model_name}_SSD_510x300_100_40_Square_${iter}.xml
elif [ "$model_name" == "MobileNet" ]; then
	iter=120000
	model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.xml
else # MobileNetV2
	iter=200000
	model=${model_name}_SSDLite_510x300_100_40_Square_1_${iter}.xml
fi

# model_name=MobileNetV2
# iter=50000
# model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.xml

cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/gtsdb/caffe/${model_name}
ls
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m $model -ODc $HOME/Documents/data/GTSDB_SSD_Classes_caffe.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU

# HETERO:FPGA,CPU
