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
	iter=30000
	model=${model_name}_SSD_510x300_100_40_Square_${iter}.caffemodel
elif [ "$model_name" == "MobileNet" ]; then
	iter=120000
	model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.caffemodel
else # MobileNetV2
	iter=200000
	model=${model_name}_SSDLite_510x300_100_40_Square_1_${iter}.caffemodel
fi

# model_name=MobileNetV2
# iter=50000
# model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.caffemodel


cd ../gtsdb/caffe/${model_name}
python3 ../../../mo.py --input_model $model --mean_values [125,127,130] --input_shape [$batch,3,300,510] --data_type $precision | grep -e abc