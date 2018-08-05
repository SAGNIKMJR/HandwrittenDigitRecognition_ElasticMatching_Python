import os
import random
import numpy as np
import bob.ip.gabor.Transform as Transform
from time import gmtime, strftime
from lib.cmdparser import parser
import lib.utils as utils

def match_n_cost(image_mask_label_array, image_transformed, no_wavelets, resized_size, match_ite, relative_weight):
    """
    This method causes the graph on an image from the evaluation set to deform elastically to minimize cost
    when matched against the stored graphs from the masking ('train') dataset
    :param image_mask_label_array: image masks from mask ('train') dataset with labels
    :param image_transformed: image from the evaluation transformation after applying the Gabor stack 
    :param resized_size: image size the data have been resized to
    :param match_ite: the number of iterations allowed for matching to a particular mask
    :param relative_weight: relative weight of the two costs (the cost from matching the vertex and the 
    						cost from deforming the edges)
    """

    initial_positions = list()

	for row in range(10):
		for col in range(10):
			initial_positions.append((row*2, col*2))

	min_cost_global = image_mask_index_min_cost_global = 0.

	# iterating through different masks stored from the mask set
	for image_mask_count in range(image_mask_label_array.shape[0]):
		print('Matching with mask no.:{}'.format(image_mask_count))
		image_mask = image_mask_label_array[image_mask_count, 0]
		image_mask_absolute = np.absolute(image_mask)
		min_cost_local_image_mask_count = float('inf')
		# running match for 'match_ite' iterations
		for ite in range(match_ite):
			new_positions = list()
			if ite>0:
				for position in initial_positions:
					direction = random.randint(1, 4)
					if direction == 1:
						if position[0]+1 > resized_size - 1:
							position = resized_size - 1, position[1]
						else:
							temp_position = position[0] + 1, position[1] 
					elif direction == 2:
						if position[0] - 1 < 0:
							temp_position = 0, position[1]
						else:
							temp_position = position[0] - 1, position[1]
					elif direction == 3:
						if position[1] + 1 > resized_size - 1:
							temp_position = position[0], resized_size - 1 
						else:
							temp_position = position[0], position[1] + 1
					elif direction == 4:
						if position[1] -1 <0:
							temp_position = position[0], 0
						else:
							temp_position = position[0], position[1] - 1
					new_positions.append(position)
			else:
				new_positions = initial_positions

			image_mask_local_movement = np.empty((no_wavelets,resized_size//2,resized_size//2),'complex128')

			row = 0
			col = 0
			for position in new_positions:
				image_mask_local_movement[:,row,col] = image_transformed[:,position[0],position[1]]
				col += 1
				if col == resized_size//2:
					row += 1
					col = 0

			sum_all_edges = 0.

			# calculating cost due to edge deformation by calculating the sum of all edge lengths
			for position_count in range(len(new_positions)):
				sum_surrounding_edge_node = 0
				if position_count - 10 < 0:
					sum_surrounding_edge_node = 0
				else:
					sum_surrounding_edge_node = ((((new_positions[position_count][0]-\
												new_positions[position_count-resized_size//2][0])**2)\
												+((new_positions[position_count][1]-\
													new_positions[position_count-10][1])**2)**0.5)-(1))**2      
				
				if (position_count+resized_size//2)>(resized_size//2)**2 - 1:
					sum_surrounding_edge_node = sum_surrounding_edge_node
				else:
					sum_surrounding_edge_node = sum_surrounding_edge_node+((((new_positions[position_count][0]-\
												positions_new[position_count+resized_size//2][0])**2)+\
												((positions_new[position_count][1]-\
												positions_new[position_count+resized_size//2][1])**2)**0.5)-(1))**2      
				
				if (position_count-1)<0 or positions[position_count][0]<positions[position_count-1][0]:
					sum_surrounding_edge_node = sum_surrounding_edge_node
				else:
					sum_surrounding_edge_node = sum_surrounding_edge_node+((((positions_new[position_new][0]-\
												positions[position_count-1][0])**2)+\
												((positions_new[position_count][1]-\
												positions_new[position_count-1][1])**2)**0.5)-(1))**2      
				
				if ((position_count+1)>(resized_size//2)**2 - 1) or (positions[position_count+1][0]<positions[position_count][0]):
					sum_surrounding_edge_node = sum_surrounding_edge_node
				else:
					sum_surrounding_edge_node = sum_surrounding_edge_node+((((positions_new[position_count][0]\
												-positions_new[position_count+1][0])**2)+\
												((positions_new[position_count][1]-\
												positions_new[position_count+1][1])**2)**0.5)-(1))**2      
				sum_all_edges += sum_surrounding_edge_node

			image_mask_local_movement_real = np.real(image_mask_local_movement)
			image_mask_local_movement_imgn = np.imag(image_mask_local_movement)
			image_mask_local_movement_abs = np.absolute(image_mask_local_movement)

			sum_all_vertices = 0.

			for row in range(resized_size//2):
				for col in range(resized_size//2):
					sum_per_vertex = square_image_mask_absolute = square_image_mask_local_movement = 0
					for wavelet_no in range(no_wavelets):
						sum_per_vertex += image_mask_absolute[row,col,wavelet_no]\
										*image_mask_local_movement_abs[row,col,wavelet_no]
						square_image_mask_absolute += (image_mask_absolute[row,col,wavelet_no])**2
						square_image_mask_local_movement += (image_mask_local_movement_abs[row,col,wavelet_no])**2

					sum_per_vertex /= (square_image_mask_absolute*square_image_mask_local_movement)**0.5
					sum_all_vertices += sum_per_vertex
		
			cost = relative_weight*sum_all_edges - sum_all_vertices

			if(cost < min_cost_local_image_mask_count or ite == 0):
				min_cost_local_image_mask_count = cost

		if(min_cost_local_image_mask_count < min_cost_global or image_mask_count == 0):
			min_cost_global = min_cost_local_image_mask_count
			image_mask_index_min_cost_global = image_mask_count

	return image_mask_label_array[image_mask_index_min_cost_global,2]

def mask(mask_data, no_wavelets, resized_size, hyperparameter_list):

    # directory traversal
    file_list=[]
    for (dirpath, dirnames, filenames) in os.walk(mask_data):
        file_list.extend(filenames)

    image_count = 0
    for image_file in file_list:
		image_file_path = os.path.join(mask_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image =np.array(image)

		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0], \
								number_of_directions=hyperparameter_list[1], \
								sigma=hyperparameter_list[3], \
								k_max=hyperparameter_list[3], \
								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)
		image_mask = np.empty((no_wavelets, resized_size//2, resized_size//2),'complex128')

		for wavelet_no in range(no_wavelets):
			for row in range(0,resized_size,2):
				for col in range(0,resized_size,2):
					image_mask[wavelet_no, row/2, col/2] = image_transformed[wavelet_no, row, col]

		if image_count == 0:
			image_mask_label_array=np.array([image_mask, label, os.path.join(match_data, image_file)])
		else:
			image_mask_label_array=np.vstack([image_mask_label_array, \
											np.array([image_mask_label_array, label, \
											os.path.join(match_data, image_file)])])

		image_count += 1

		return image_mask_label_array

def eval(eval_data, mask_data, resized_size, image_mask_label_array, no_wavelets, log, hyperparameter_list, match_ite, \
		relative_weight):

    # directory traversal
    file_list=[]
    for (dirpath, dirnames, filenames) in os.walk(eval_data):
        file_list.extend(filenames)

    image_count = 0
    image_count_match = 0

    for image_file in file_list:
		image_file_path = os.path.join(eval_data, image_file)
		image = Image.open(image_file_path)
		label = image_file_path[len(image_file_path):len(image_file_path)+1]
		image =np.array(image)

   		gabor_wavelets = Transform(number_of_scales=hyperparameter_list[0], \
   								number_of_directions=hyperparameter_list[1], \
   								sigma=hyperparameter_list[3], \
   								k_max=hyperparameter_list[3], \
   								k_fac=hyperparameter_list[4])
		image_transformed = gabor_wavelets.transform(image)

		match_file_path = match_n_cost(image_mask_label_array, image_transformed, no_wavelets, resized_size, \
									match_ite, relative_weight)
		match_label = match_file_path[len(match_file_path):len(match_file_path)+1]

		if match_label == label:
			image_count_match += 1
			image_count += 1
		else:
			image_count += 1
			print("Iteration No.:{}".format(image_count))
			print("Eval Image File:{}".format(image_file_path))
			print("Wrong Match Image File:{}".format(match_file_path))
			log.write("Iteration No.:{}".format(image_count))
			log.write("Eval Image File:{}".format(image_file_path))
			log.write("Wrong Match Image File:{}".format(match_file_path))
		if image_count%int(args.print_freq) == 0:
			print("Iteration No.:{}".format(image_count))\
			print("Current Evaluation Accuracy:{}".format((image_count_match*100.)/image_count))
			log.write("Iteration No.:{}".format(image_count))
			log.write("Current Evaluation Accuracy:{}".format((image_count_match*100.)/image_count))

		return (image_count_match*100.)/image_count

def main():
	save_path = './runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	log_file = os.path.join(save_path, "stdout")
	log = open(log_file, "a")

	args = parser.parse_args()
	print("Command line options:")
	for arg in vars(args):
		print(arg, getattr(args, arg))
		log.write(arg + ':' + str(getattr(args, arg)) + '\n')

	utils.extract(args.dataset, args.raw_data)
	utils.resize(args.mask_data, args.resized_size)
	utils.resize(args.eval_data, args.resized_size)
	utils.deskew(args.image_mask_countdata)
	utils.deskew(args.eval_data)

	hyperparameter_list = [args.no_scales, args.no_directions, args.sigma,\
						 args.frequency_max, args.frequency_factor]
	image_mask_label_array = mask(args.mask_data, args.no_scales*args.no_directions, \
								args.resized_size, hyperparameter_list)
	eval_accuracy = eval(args.eval_data, args.mask_data, args.resized_size, args.image_mask_label_array, \
						args.no_scales*args.no_directions, log, hyperparameter_list, args.match_ite, \
						args.relative_weight)

	print("Final Evaluation Accuracy:{eval_accuracy:.3f}".format(eval_accuracy))
	log.write("Final Evaluation Accuracy:{eval_accuracy:.3f}".format(eval_accuracy))

	log.close()

if __name__ == '__main__':
	main()


