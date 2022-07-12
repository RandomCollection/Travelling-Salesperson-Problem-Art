# MODULE:       tsp_art.py
# VERSION:      1.0
# DIRECTORY:    <masked>
# DATE:         2022-07-12
# AUTHOR:       RandomCollection
# DESCRIPTION:  See https://github.com/RandomCollection/Travelling-Salesperson-Problem-Art.

# LIBRARIES ############################################################################################################

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy_numpy import solve_tsp


# FUNCTIONS ############################################################################################################

def normalise_figsize(img: Image.Image) -> (float, float):
	width, height = img.size
	total = width + height
	width_norm = width / total * 10
	height_norm = height / total * 10
	return width_norm, height_norm


def show_image(img: Image.Image, coordinates: np.array, plot_type: str, save: str = None):
	plt.figure(figsize=normalise_figsize(img))
	if plot_type == "scatter":
		plt.scatter(x=[x[1] for x in coordinates], y=[x[0] for x in coordinates], s=1, color="black")
	elif plot_type == "line":
		plt.plot([x[1] for x in coordinates], [x[0] for x in coordinates], lw=1, color="black")
	else:
		raise ValueError(f"plot_type '{plot_type}' is not defined in function 'show_image'")
	plt.axis("off")
	plt.tight_layout(pad=-1.5)
	if save is not None:
		plt.savefig(fname=save)


# MAIN FUNCTION ########################################################################################################

def tsp_art():
	# GET REPOSITORIES -------------------------------------------------------------------------------------------------

	repository_in = input(r"Enter the repository of the image including name and type such as 'C:\...\image_in.png':")
	repository_out = input(r"Enter the repository of the image including name and type such as 'C:\...\image_out.png':")

	# IMAGE IMPORT -----------------------------------------------------------------------------------------------------

	img = Image.open(fp=repository_in)

	# IMAGE MANIPULATION -----------------------------------------------------------------------------------------------

	img_bw = img.convert(mode="1", dither=Image.NONE)
	img_bw_array = np.array(img_bw, dtype=int)
	black_indices = np.argwhere(a=img_bw_array == 0)
	chosen_black_indices = black_indices[np.random.choice(a=black_indices.shape[0], size=1_000, replace=False)]
	show_image(img=img, coordinates=chosen_black_indices, plot_type="scatter")
	distances = pdist(X=chosen_black_indices)
	distance_matrix = squareform(X=distances)
	optimized_path = solve_tsp(distances=distance_matrix)
	optimized_path_points = [chosen_black_indices[x] for x in optimized_path]

	# IMAGE EXPORT -----------------------------------------------------------------------------------------------------

	show_image(img=img, coordinates=optimized_path_points, plot_type="line", save=repository_out)


# MAIN #################################################################################################################

if __name__ == "__main__":
	tsp_art()

# END ##################################################################################################################
