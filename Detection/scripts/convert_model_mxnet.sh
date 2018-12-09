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
# ./install_prerequisites.sh mxnet venv
source ../venv/bin/activate

# cd ../ssd/mxnet/vgg16_ssd_300_voc0712_trainval
# python3 ../../../mo_mxnet.py --input_model ssd_300-0000.params --input_shape [1,3,300,300]

# GTSDB
if [ "$model_name" == "VGG" ]; then
	model=deploy_gtsdb_ssd_vgg16_reduced_300_510-0102.params
elif [ "$model_name" == "MobileNet" ]; then
	model=deploy_gtsdb_ssd_mobilenet_v1_300_510-0102.params
else # MobileNetV2
	model=deploy_gtsdb_ssdlite_mobilenet_v2_300_510-0204.params
fi

# model=deploy_gtsdb_ssd_mobilenet_v2_300_510-0102.params
cd ../gtsdb/mxnet
python3 ../../mo_mxnet.py --input_model $model --mean_values [125,127,130] --input_shape [$batch,3,300,510] --data_type $precision --reverse_input_channels | grep -e abc

# GTSRB
# model=deploy_idsia_mxnet_GT_48by48_3-0025.params
# size=32
# model=deploy_resnet-20_mxnet_GT_${size}by${size}_3-0025.params
# size=64
# model=deploy_resnet-32_mxnet_GT_${size}by${size}_3-0025.params
# cd ../classification/mxnet
# python3 ../../mo_mxnet.py --input_model $model --input_shape [32,3,$size,$size] --data_type float --scale 255 --input data