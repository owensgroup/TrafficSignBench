fp=$1
model_name=$2
if [ -z "$fp" ]
then
	fp="FP11"
fi
if [ -z "$model_name" ]
then
	model_name="ResNet20"
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
elif [ "$model_name" == "IDSIA" ]; then # IDSIA
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_VGG_Generic.aocx
else # Others
	aocl program acl0 $OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
	export DLA_AOCX=$OPENVINO_ROOTDIR/bitstreams/a10_devkit_bitstreams/4-0_A10DK_FP16_MobileNet_ResNet_SqueezeNet_Clamp.aocx
fi

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