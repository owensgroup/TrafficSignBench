cd $DATASET_ROOT

# Get GTSRB dataset
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip

# Unzip dataset
mkdir GTSRB
tar -xf GTSRB_Final_Training_Images.zip
tar -xf GTSRB_Final_Test_Images.zip
tar -xf GTSRB_Final_Test_GT.zip -C ./GTSRB/