import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (21,8)

count = 0
legends = ["FPGA-FP11-half", "FPGA-FP11-float", "FPGA-FP16-half", "FPGA-FP16-float", "GPU"]
# xlabels = ["VGG-SSD", "MobileNet-V1-SSD", "MobileNet-V2-SSDLite"]
xlabels = ["VGG", "Mv1", "Mv2", "R18", "R50", "Sqz"]
data = {"Caffe": [], "MXNet": [], "Tensorflow": []}
fws = ["Caffe", "MXNet", "Tensorflow"]
colors = ["#0071c5", "#3393d8", "#66b5eb", "#99d7fe", "#76b900"]
fontsize = 21

with open("./pe.csv", 'r') as f:
	lines = f.readlines()
		
	while(count < 18):
		line = lines[count]
		splitted = [1.0/float(x) for x in line.strip().split(',')]
		if count >= 0 and count < 6:
			data["Caffe"].append(splitted)
		elif count >= 6 and count < 12:
			data["MXNet"].append(splitted)
		else:
			data["Tensorflow"].append(splitted)
		count += 1

fig = plt.figure()
bars = []
for idx, fw in enumerate(fws):
	num_model = 6
	ax = plt.subplot(1,3,idx+1)
	ax.set_title(fw, x=0.5, y=1, fontsize=fontsize)
	for i in range(5):
		d = []
		for j in range(num_model):
			d.append(data[fw][j][i])
		tmp = plt.bar([-0.2 + i*0.1 + x for x in range(num_model)], d, 0.1, color=colors[i])
		if fw == "Tensorflow":
			bars.append(tmp)
	plt.xticks([x-0.1 for x in range(num_model)], xlabels[(6-num_model):], rotation=8, fontsize=18)
	plt.yticks(range(12), fontsize=18)
	plt.ylim([-0.2, num_model-0.2])
	plt.ylim([0, 11.2])
	plt.grid(axis='y', linestyle='dashdot')
fig.legend(bars, legends, loc=(0.84, 0.67), fontsize=16)
plt.tight_layout(pad=4.5, w_pad=2, h_pad=1)
fig.text(0.007, 0.5, 'img/J', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.03, 'Model', va='center', rotation='horizontal', fontsize=fontsize)
fig.savefig("/Users/moderato/Downloads/IV-paper/Special Issue/power_efficiency.png", dpi=fig.dpi)
plt.show()