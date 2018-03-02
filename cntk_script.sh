cnn="idsia"
size_xy=48
dataset="GT"
epoch=25
batch=64
process=3 # 0 for none, 1 for 1-sigma, 2 for 2-sigma, 3 for clahe
print=0

source deactivate
source activate cntk

python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

cnn="resnet-20"
size_xy=64

python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

size_xy=48

python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

size_xy=32

python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

# cnn="resnet-32"

# python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

# size_xy=48

# python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print