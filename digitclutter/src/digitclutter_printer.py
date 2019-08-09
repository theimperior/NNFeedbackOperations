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
n_digits = [3, 4, 5]
resolution = [(512, 512), (32, 32)]
font_set = ['arial-bold']

for res in resolution:
	for n_digit in n_digits:
		clutterlist = read_image_set("../digitclutter/trainset/csv/5000_" + str(n_digit) + "digits10.csv")
		clutterlist = clutterlist[sampleStart:sampleEnd]
		clutterlist = io.name_files('./digitclutter/', clutter_list=clutterlist)
		#saving the images only with the digits
		for i, cl in enumerate(clutterlist):
			cl.render_occlusion()

		fname_list = [cl.fname for cl in clutterlist]
		images_dict = io.save_images_as_mat(abspath("./" + str(res[0]) + "_" + str(n_digit) + "digits" + ".mat"), 
										clutterlist, res, fname_list=fname_list, delete_bmps=True, overwrite_wdir=True)
				
		for i in range(0,images_dict['images'].shape[0]):
			png.from_array(images_dict['images'][i], 'L').save(str(res[0]) + "_" + str(n_digit) + "digits_" + str(i) + "_target_" + str(images_dict['targets'][i]) + ".png")