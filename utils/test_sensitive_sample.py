# from model import MlpMNIST
from utils.objectives import MLP_MNIST
from torchvision.datasets import MNIST
import torch
import os
import cv2
import copy
import glob
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(device)
model.load_state_dict(torch.load("models_and_data/pretrained_models/trained_MNIST_acc_0.93.pth"))
model.eval()

N = range(1,11)
RNUM = 200
N_TAMPED_MODELS = 1000
def evaluate_detection():
	'cEI', 'UCBC', 'ADMMBO'
	detection_rates = {"Neural-CBO": [0]*len(N), 
					   "ConfigOpt": [0]*len(N),
					   "cEI": [0]*len(N),
					   "UCBC": [0]*len(N),
					   "ADMMBO": [0]*len(N)} 


	for j in tqdm(range(N_TAMPED_MODELS)):
		tamped_model_path = 'models_and_data/pretrained_models/tamped_models/tamped_mnist.{j}.pth'
		if os.path.isfile(f'')==False:
			tamped_model = copy.deepcopy(model)
			with torch.no_grad():
				for i in range(tamped_model.layer1.weight.shape[0]):
						sz = tamped_model.layer1.weight[i].shape[0]
						tamped_model.layer1.weight[i] += torch.normal(torch.zeros(sz), torch.full([sz], 0.04)).to(device)
					
				for i in range(tamped_model.layer2.weight.shape[0]):
						sz = tamped_model.layer2.weight[i].shape[0]
						tamped_model.layer2.weight[i] += torch.normal(torch.zeros(sz), torch.full([sz], 0.04)).to(device)
			tamped_model.eval()
			torch.save(tamped_model.state_dict(), tamped_model_path)
		else:
			tamped_model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(device)
			tamped_model.load_state_dict(torch.load(tamped_model_path))
		
		algs = ["Neural-CBO", 'ConfigOpt', 'UCBC']
		for n in N:		
			for alg in algs:
				
				images = glob.glob(f'asset/sensitive_sample_best/{alg}_{RNUM}/*')
				images = random.choices(images, k=n)
				
				for img_name in images:
					img = cv2.imread(img_name,0)
					img = torch.tensor(img).unsqueeze(1).float().div(255).sub_(0.1307).div_(0.3081).to(device)

					if torch.argmax(tamped_model(img)) != torch.argmax(model(img)):
						if alg in detection_rates:
							detection_rates[alg][n-1]+=1
							break
		
	print(detection_rates)



def plot(detection_rates):
	colors_map = {'NeuralBO': 'red', 'NeuralBO_static': 'brown', 'NeuralTS': 'lightsalmon', 'NeuralGreedy': 'blue', 'DNGO':'forestgreen', 
					'RF': 'black', 'GPEI': 'grey', 'GPUCB':'purple', 'GPTS': 'gold'}
	markers = {'NeuralBO':"*", 'NeuralBO_static':">", 'NeuralGreedy': "v", "GPEI": "^", "GPTS":"s",  'GPUCB': "o",
				"RF":"+", "DNGO":"x"}
	algs = ["NeuralBO", "NeuralGreedy", "DNGO", "GPTS", 'GPUCB', "GPEI", "RF"]
	for alg in algs:
		acc = detection_rates[alg]
		plt.plot(N, np.array(acc)/N_TAMPED_MODELS, label = alg, color = colors_map[alg],  marker=markers[alg])


	# plt.title(f"Sensitive Sample Detection rates", fontsize=15)
	fig = plt.gcf()
	leg = plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.5),loc='lower center', ncol=4)
	for legobj in leg.legendHandles:
		legobj.set_linewidth(3.0)
	plt.ylabel("Detection Rates", fontsize=15)
	plt.xlabel("Number of Samples", fontsize=15)
	plt.grid()

	fig.tight_layout() 
	if os.path.isdir("figures") ==False:
		os.makedirs("figures")
	plt.savefig(f'figures/sensitive_sample_detection_rate.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.savefig(f'figures/sensitive_sample_detection_rate.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.clf()

if __name__ =='__main__':
	evaluate_detection()
	
	# plot(detection_rates)

	






