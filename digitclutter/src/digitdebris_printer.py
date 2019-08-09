import sys
sys.path.append('../')
import platform
from os.path import abspath
from digitclutter import generate, io
from scipy.io import savemat
from digitclutter.io import read_image_set
import png

# png.from_array([[255, 0, 0, 255],[0, 255, 255, 0]], 'L').save("small_smiley.png")
"""
Reads a csv file from disk and saves the full resolution images 
"""
n_samples = 5000
sampleStart = 1234
sampleEnd = sampleStart + 10
n_debris_sets = [[10, 11], [30, 31], [50, 51]]
resolution = [(512, 512), (32, 32)]
font_set = ['arial-bold']

for res in resolution:
	for n_debris in n_debris_sets:
		clutterlist = read_image_set("../digitdebris/trainset/csv/5000_" + str(n_debris[0]) + "debris10.csv")
		clutterlist = clutterlist[sampleStart:sampleEnd]
		clutterlist = io.name_files('./digitdebris/', clutter_list=clutterlist)
		#saving the images only with the digits
		for i, cl in enumerate(clutterlist):
			cl.render_occlusion()

		fname_list = [cl.fname for cl in clutterlist]
		images_dict = io.save_images_as_mat(abspath('./digitdebris/' + str(n_samples) + '_' + str(n_debris[0]) + '_no_debris' + str(1) + '.mat'), 
											clutterlist, res, fname_list=fname_list, delete_bmps=False, overwrite_wdir=True)

		# Make debris 
		if res == (512, 512):
			debris_size=[96,141]
		else:
			debris_size=[6,9]
			
		debris_array = generate.make_debris(len(clutterlist), n_debris=n_debris, font_set=font_set, image_save_size=res, debris_size=debris_size)
		images_with_debris = generate.add_debris(images_dict['images'], debris_array)

		for i in range(0,images_with_debris.shape[0]):
			png.from_array(images_with_debris[i,:,:,:], 'L').save(str(res[0]) + "_" + str(n_debris[0]) + "debris_" + str(i) + "_target_" + str(images_dict['targets'][i][0]) + ".png")