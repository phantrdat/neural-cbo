import torch
import time
import math
def generate_uniform_points_on_sphere(num_points, dim):
	# Step 1: Generate random points from a standard normal distribution
	random_points = torch.randn(num_points, dim)
	
	# Step 2: Normalize these points to lie on the unit sphere
	norms = torch.norm(random_points, p=2, dim=1, keepdim=True)
	points_on_sphere = random_points / norms
	
	return points_on_sphere

def random_seed(T=0):
	frac, whole = math.modf(time.time())
	try:
		seed = int(whole/(10000*frac)) + T
		torch.manual_seed(seed)
	except:
		torch.manual_seed(1111)