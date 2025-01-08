import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import imageio
import os
import shutil
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='results/Simionescu_DIM_2_ITERS_100/Constrained_NeuralBO_LVS/Constrained_NeuralBO_LVS_Simionescu_dim2.1.pkl', help='Result File')
parser.add_argument('--a', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Draw all files')
parsed, unknown = parser.parse_known_args()
args = parser.parse_args()

def Ackley(X,Y):
	c0 = 1 - (np.linalg.norm(np.array([(X-1), (Y-1)]), axis=0) - 5.5)**2 
	c1 = np.maximum(np.abs(X), np.abs(Y))**2 - 9
	return [c0, c1]
def Gardner(X,Y):
	return [np.sin(X) * np.sin(Y) + 0.95]

def Gramacy(X,Y):
	c0 = - 0.5*np.sin(2*np.pi*(X**2 - 2*Y)) - X - 2*Y + 1.5
	c1 = X**2 + Y**2 - 1.5
	return  [c0, c1]

def Branin(X,Y):
	return [(X-2.5)**2 + (Y-7.5)**2 - 50] 
def Simionescu(X,Y):
	rT = 1
	rS = 0.2 
	n = 8
	c0 =  X**2 + Y**2 - (rT + rS*np.cos(n*np.arctan (X/Y) ))**2
	return [c0]
def Gomez_and_Levy(X,Y):
	return [-np.sin(4*np.pi*X) + 2*np.sin(2*np.pi*Y)**2 - 1.5]
func = {"Gardner": Gardner, 
		'Gramacy': Gramacy, 
		'BraninHoo': Branin, 
		'Simionescu': Simionescu,
		'Gomez_and_Levy': Gomez_and_Levy,
		'Ackley': Ackley}
def create_gif(image_folder, output_file, duration=0.5):
		"""
		Create a GIF from a folder containing images.
		
		Parameters:
				image_folder (str): Path to the folder containing images.
				output_file (str): Output file name (with .gif extension).
				duration (float): Time duration (in seconds) between frames.
		"""
		images = []
		
		# Iterate through all images in the folder
		for filename in sorted(os.listdir(image_folder)):
				if filename.endswith('.png') or filename.endswith('.jpg'):
						image_path = os.path.join(image_folder, filename)
						images.append(Image.open(image_path))
		
		# Save the images as a GIF
		imageio.mimsave(output_file, images, duration=duration)


def plot_feasible_regions(func_name, min_x, max_x, min_y, max_y):
	x = np.linspace(min_x, max_x, 100)  # 2D array
	y = np.linspace(min_y, max_y, 100)     # 1D array
	X, Y = np.meshgrid(x, y)      # Creating grid from 1D arrays
	constraints = func[func_name](X,Y)
	feasible_masks =  [constraints[i] <=0  for i in range(len(constraints))]
	
	overall_feasible_mask = np.all(feasible_masks, axis=0)
	overall_infeasible_mask = ~overall_feasible_mask
	# for i, R in enumerate(feasible_regions):
		
	plt.contourf(X,Y, overall_feasible_mask, levels=[-0.5, 0.5], colors='white', alpha=0.5)
	plt.contourf(X, Y, overall_infeasible_mask, colors='green',  levels=[-0.5, 0.5])
	
	for i, c in enumerate(constraints):
		plt.contour(X, Y, c, levels=[0], colors='blue', linestyles='dashed')

	plt.xlabel('X1')
	plt.ylabel('x2')
	plt.xlim(left=min_x)
	plt.ylim(bottom=min_y)
	plt.title(f'2D Constrained Optimization - {func_name}')
	plt.tight_layout()
	plt.savefig(f"{func_name}_feasible.png")

def plot_2d_constrained_optimization(optimal_pkl_file, root_dir='visualization/'):
	
	# temp_dir = 'temp/'
	
	# if os.path.isdir(temp_dir)==False:
	# 	os.mkdir(temp_dir)
	if os.path.isdir(root_dir)==False:
		os.mkdir(root_dir)
	
	D = pkl.load(
			open(optimal_pkl_file,'rb'), encoding='utf-8')
	if "u_directions" in D:
		directions = D["u_directions"]
	if "X0" in D:
		LX0 = D['X0']
	min_x, min_y = D['function_properties']['min'].cpu().numpy()
	max_x, max_y = D['function_properties']['max'].cpu().numpy()
	func_name = D['function_name']
	
	# Generate sample data
	x = np.linspace(min_x, max_x, 100)  # 2D array
	y = np.linspace(min_y, max_y, 100)     # 1D array
	X, Y = np.meshgrid(x, y)      # Creating grid from 1D arrays

	   # Sample 2D data for the colormap
	# infeasible_pts_idx = [np.where(Z > cb)[0] for cb in constraint_boundaries] 
	# infeasible_idx = list(set.intersection(*map(set,infeasible_pts_idx)))
	# infeasible_idx = sorted([f for f in infeasible_idx])
	# Z[infeasible_pts_idx] = 0

	# Create colormap plot
	plt.figure(figsize=(8, 6))
	
	constraints = func[func_name](X,Y)
	feasible_masks =  [constraints[i] <=0  for i in range(len(constraints))]
	
	overall_feasible_mask = np.all(feasible_masks, axis=0)
	overall_infeasible_mask = ~overall_feasible_mask
	# for i, R in enumerate(feasible_regions):
		
	plt.contourf(X,Y, overall_feasible_mask, levels=[-0.5, 0.5], colors='white', alpha=0.5)
	plt.contourf(X, Y, overall_infeasible_mask, colors='green',  levels=[-0.5, 0.5])
	
	for i, c in enumerate(constraints):
		plt.contour(X, Y, c, levels=[0], colors='blue', linestyles='dashed')

	plt.xlabel('X1')
	plt.ylabel('x2')
	plt.xlim(left=min_x)
	plt.ylim(bottom=min_y)
	plt.title(f'2D Constrained Optimization - {func_name}')
	plt.tight_layout()
	

	optimal_points = D['X_train'][-100:].numpy()
	optimal_values = D['optimal_values'][-100:]
	constraint_values = D['constraint_values'][-100:]
	is_feasible =  [(torch.FloatTensor(v)<=0).all() for v in constraint_values]
	feasible_pts = optimal_points[is_feasible]
	feasible_values = np.array(optimal_values)[is_feasible]
	minimal_value = None
	if feasible_values.shape[0] !=0:
		# print("feasible found:", feasible_pts)

		min_idx = np.argmin(feasible_values)
		minimal_value = feasible_values[min_idx]
		minimum = feasible_pts[min_idx]
		print(f"feasible minimum {minimum}, min feasible value {minimal_value}")

	X1 = optimal_points[:,0]
	X2 = optimal_points[:,1]

	# gif_id = optimal_pkl_file.split('.')[1]
	
	# x_opt = [0.8, 0.7]
	# plt.plot(x_opt[0], x_opt[1], 'x', color='black')
	
	for k, (x1, x2) in enumerate(zip(X1, X2)):
			if optimal_values[k] == minimal_value:
				marker = "*"
				color = "red"
				marker_size = 30
			else: 
				marker = '.'
				color = "purple"
				marker_size = 5
			plt.scatter(x1, x2, s=marker_size, color=color, marker=marker)
			# plt.quiver(LX0[k][0], LX0[k][1], directions[k][0], directions[k][0], angles='xy', scale_units='xy', color='r', width=0.001)
			plt.plot()
			# plt.savefig(f"gif_figs/{k:03d}.png", dpi=300)
			if k== len(X1)-1:
				
				alg_dir = os.path.basename(optimal_pkl_file).split('.')[0]
				output_dir = os.path.join(root_dir, alg_dir)
				if os.path.isdir(output_dir)==False:
					os.makedirs(output_dir)
				
				fn = os.path.basename(optimal_pkl_file).replace('pkl', 'png')
				figure_path = os.path.join(output_dir, fn)
				
				plt.savefig(figure_path, dpi=300)
	# 
	# save last figures 
	
	# gif_id = optimal_pkl_file.split('.')[1]
	# create_gif("gif_figs/",f'{func_name}_c_{gif_id}.gif', duration=300)
	# shutil.rmtree('gif_figs')
	
	plt.clf()


if __name__ == '__main__':

	# plot_2d_constrained_optimization()
	# plot_feasible_regions("Ackley", min_x=-3, max_x=5, min_y=-3, max_y=5)

	if args.a == False:
		plot_2d_constrained_optimization(optimal_pkl_file=args.i)
	else:
		import glob
		pkl_files = sorted(glob.glob(os.path.join(args.i, '*')))
		
		for f in pkl_files:
			# print(f)
			plot_2d_constrained_optimization(f)
		