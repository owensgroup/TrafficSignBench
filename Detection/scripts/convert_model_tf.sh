model_name=$1
batch=$2
precision=$3
if [ -z "$model_name" ]
then
	model_name="ResNet20"
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

# Use R3 for now as R4 throws a bug
cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3/install_prerequisites
./install_prerequisites.sh tf venv
source ../venv/bin/activate

# GTSRB
cd ../classification/tf
if [ "$model_name" == "IDSIA" ]; then
	model=idsia_48by48.pb
	python3 ../../mo_tf.py --input_model $model -b $batch --input=idsia/input_placeholder --output=idsia/probs --data_type $precision --scale 255 | grep -e abc
elif [ "$model_name" == "ResNet20" ]; then
	model=resnet-20_32by32.pb
	python3 ../../mo_tf.py --input_model $model -b $batch --input=resnet-20/input_placeholder --output=resnet-20/probs --data_type $precision --scale 255 # | grep -e abc
else # ResNet32
	model=resnet-32_64by64.pb
	python3 ../../mo_tf.py --input_model $model -b $batch --input=resnet-32/input_placeholder --output=resnet-32/probs --data_type $precision --scale 255 # | grep -e abc
fi
