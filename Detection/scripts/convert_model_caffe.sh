cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/install_prerequisites
# ./install_prerequisites.sh caffe venv
source ../venv/bin/activate

# model_name=VGG
# iter=30000
# model=${model_name}_SSD_510x300_100_40_Square_${iter}.caffemodel

model_name=MobileNet
iter=80000
model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.caffemodel

# model_name=MobileNetV2
# iter=50000
# model=${model_name}_SSD_510x300_100_40_Square_1_${iter}.caffemodel

# model_name=MobileNetV2
# iter=200000
# model=${model_name}_SSDLite_510x300_100_40_Square_1_${iter}.caffemodel

cd ../gtsdb/caffe/${model_name}
python3 ../../../mo.py --input_model $model --mean_values [125,127,130] --input_shape [4,3,300,510] --data_type half