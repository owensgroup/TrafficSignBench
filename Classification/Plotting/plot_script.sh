cnn="idsia" # Options: "idsia", "resnet-20", "resnet-32"
size_xy=48
dataset="GT"
preprocessing="clahe"
title=1

# Plot loss/accuracy
python Plot_Loss_Accuracy.py --root $DATASET_ROOT --network_type $cnn --resize_side $size_xy --dataset $dataset --preprocessing $preprocessing --devices gpu

# Plot training time versus input sizes
cnn="resnet-20" # Options: "resnet-20", "resnet-32"
python Plot_Time_Size.py --root $DATASET_ROOT --network_type $cnn --dataset $dataset --preprocessing $preprocessing --devices gpu --title $title

# Plot training time versus models
python Plot_Time_Model.py --root $DATASET_ROOT --dataset $dataset --preprocessing $preprocessing --devices gpu --title $title