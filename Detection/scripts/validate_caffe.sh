fp=$1
model_name=$2
if [ -z "$fp" ]
then
	fp="FP16"
fi
if [ -z "$model_name" ]
then
	model_name="VGG"
fi

# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# # Program the board
# # R4
# aocx="4-0_A10DK_FP16_VGG_Generic.aocx"
# if [ "$fp" == "FP11" ]; then
# 	aocx="4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx"
# # FP16
# elif [ "$model_name" == "VGG" ]; then # VGG
# 	aocx="4-0_A10DK_FP16_VGG_Generic.aocx"
# else # Non-VGG
# 	aocx="4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx"
# fi
# R5
aocx="5-0_A10DK_FP16_Generic.aocx"
if [ "$fp" == "FP11" ]; then
	if [ "$model_name" == "VGG" ]; then
		aocx="5-0_A10DK_FP11_VGG.aocx"
	else
		aocx="5-0_A10DK_FP11_MobileNet_Clamp.aocx"
	fi
# FP16
else
	if [ "$model_name" == "VGG" ]; then
		aocx="5-0_A10DK_FP16_SqueezeNet_VGG.aocx"
	else
		aocx="5-0_A10DK_FP16_MobileNet_Clamp.aocx"
	fi
fi

aocl program acl0 "${OPENVINO_ROOTDIR}/bitstreams/a10_devkit_bitstreams/${aocx}"
export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/$aocx

printf $"Args: ${fp}, ${model_name}, ${aocx}\n"

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
$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m $model -ODc $HOME/Documents/data/GTSDB_SSD_Classes_caffe.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU

# HETERO:FPGA,CPU
