cd $DATA

# Get GTSRB dataset
count=0
for file in GTSRB_Final_Training_Images.zip GTSRB_Final_Test_Images.zip GTSRB_Final_Test_GT.zip; do
	count=$((${count}+1))
	if [ ! -f $file ]; then
		wget http://benchmark.ini.rub.de/Dataset/${file}
		# Unzip dataset
		if [ $count -eq 3 ]; then
			unzip $file -d ./GTSRB
		else
			unzip $file
		fi
	fi
done