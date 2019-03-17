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
if [ "$model_name" == "VGG16" ] || [ "$model_name" == "MobileNet" ] || [ "$model_name" == "MobileNetV2" ] || [ "$model_name" == "ResNet18" ] || [ "$model_name" == "ResNet50" ] || [ "$model_name" == "SqueezeNet11" ] ; then 
	type="detection"
else
	type="classification"
fi

# Set up variables
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

# Program the board
# # R4
# aocx="4-0_A10DK_FP16_VGG_Generic.aocx"
# if [ "$fp" == "FP11" ]; then
# 	aocx="4-0_A10DK_FP11_MobileNet_ResNet_VGG_Clamp.aocx"
# else # FP16
# 	aocx="4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx"
# fi
# R5
aocx="5-0_A10DK_FP16_Generic.aocx"
if [ "$fp" == "FP11" ]; then
	if [ "$model_name" == "VGG16" ]; then
		aocx="5-0_A10DK_FP11_VGG.aocx"
	elif [ "$model_name" == "MobileNet" ] || [ "$model_name" == "MobileNetV2" ]; then
		aocx="5-0_A10DK_FP11_MobileNet_Clamp.aocx"
	elif [ "$model_name" == "ResNet18" ] || [ "$model_name" == "ResNet50" ]; then
		aocx="5-0_A10DK_FP11_ResNet18.aocx"
	else # SqueezeNet v1.1
		aocx="5-0_A10DK_FP11_SqueezeNet.aocx"
	fi
# FP16
else
	if [ "$model_name" == "VGG16" ] || [ "$model_name" == "SqueezeNet11" ]; then
		aocx="5-0_A10DK_FP16_SqueezeNet_VGG.aocx"
	elif [ "$model_name" == "MobileNet" ] || [ "$model_name" == "MobileNetV2" ]; then
		aocx="5-0_A10DK_FP16_MobileNet_Clamp.aocx"
	else # ResNet18/50
		aocx="5-0_A10DK_FP16_ResNet_TinyYolo.aocx"
	fi
fi

aocl program acl0 "${OPENVINO_ROOTDIR}/bitstreams/a10_devkit_bitstreams/${aocx}"
export DLA_AOCX="$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/${aocx}"

printf $"Args: ${fp}, ${model_name}, ${aocx}\n"

if [ "$type" == "detection" ]; then
	cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3/gtsdb/tf/${model_name}
	$OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples/build/intel64/Release/validation_app -t OD -ODa $HOME/Documents/data/GTSDBdevkit/GTSDB/Annotations/test -i $HOME/Documents/data/GTSDBdevkit -m frozen_inference_graph.xml -ODc $HOME/Documents/data/GTSDB_SSD_Classes_tf.txt -ODsubdir JPEGImages/test -d HETERO:FPGA,CPU
else # classification
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
fi

