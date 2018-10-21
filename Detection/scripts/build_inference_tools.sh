# Make inference examples
cd $OPENVINO_ROOTDIR/bin
source setupvars.sh

cd $OPENVINO_ROOTDIR/deployment_tools/inference_engine/samples
mkdir build
cd build
cmake ..
make -j8

# ./build_samples.sh