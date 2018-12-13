fp=$1
model_name=$2
if [ -z "$fp" ]
then
	fp="FP16"
fi
if [ -z "$model_name" ]
then
	# model_name="ResNet20"
	model_name="MobileNetV2"
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
elif [ "$model_name" == "IDSIA" ] || [ "$model_name" == "VGG" ]; then # IDSIA / VGG
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
else # Others
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
fi

# # GTSDB
# if [ "$model_name" == "VGG" ]; then
# 	iter=30000
# 	model=${model_name}_SSD_510x300_100_40_Square_${iter}.xml
# elif [ "$model_name" == "MobileNet" ]; then
# 	iter=120000
# 	model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.xml
# else # MobileNetV2
# 	iter=200000
# 	model=${model_name}_SSDLite_510x300_100_40_Square_1_${iter}.xml
# fi


# cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3/gtsdb/tf/${model_name}
# $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m frozen_inference_graph.xml -ODc $HOME/Documents/data/GTSDB_SSD_Classes_tf.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU

# GTSRB
if [ "$model_name" == "IDSIA" ]; then
	size=48
	model=idsia_${size}by${size}.xml
elif [ "$model_name" == "ResNet20" ]; then
	size=32
	model=resnet-20_${size}by${size}.xml
else # ResNet32
	size=64
	model=resnet-32_${size}by${size}.xml
fi

# Use R3 for now as R4 throws a bug
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3/classification/tf
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t C -i $HOME/Documents/data/GTSRB/Final_Test/pp_imgs_clahe_${size}_${size}/labels.txt -m $model -d HETERO:FPGA,CPU