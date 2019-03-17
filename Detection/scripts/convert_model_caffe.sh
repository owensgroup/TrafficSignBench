model_name=$1
batch=$2
precision=$3
if [ -z "$model_name" ]
then
	model_name="VGG"
fi
if [ -z "$batch" ]
then
	batch=1
fi
if [ -z "$precision" ]
then
	precision="half"
fi
echo "**********"
printf $"Model: $model_name\nbatch = ${batch}, precision = ${precision}\n"
echo "**********"

cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/install_prerequisites
# ./install_prerequisites.sh caffe venv
source ../venv/bin/activate

if [ "$model_name" == "VGG" ]; then
	model=${model_name}_SSD_510x300_100_40_Square_30000.caffemodel
elif [ "$model_name" == "MobileNet" ]; then
	model=${model_name}_SSD_510x300_100_40_Square_1_120000.caffemodel
elif [ "$model_name" == "MobileNetV2" ]; then
	model=${model_name}_SSDLite_510x300_100_40_Square_1_200000.caffemodel
elif [ "$model_name" == "ResNet18" ] || [ "$model_name" == "ResNet50" ]; then
	model=${model_name}_SSD_510x300_100_40_Square_1_80000.caffemodel
else # SqueezeNet v1.1
	model=${model_name}_SSD_510x300_100_40_Square_1_10000.caffemodel
fi

# model_name=MobileNetV2
# model=${model_name}_SSD_510x300_100_40_Square_1_50000.caffemodel

cd ../gtsdb/caffe/${model_name}
python3 ../../../mo.py --input_model $model --mean_values [125,127,130] --input_shape [$batch,3,300,510] --data_type $precision | grep -e abc