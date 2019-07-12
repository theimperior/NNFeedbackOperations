import sys
sys.path.append('../')
import platform
from os.path import abspath
from digitclutter import generate, io
from scipy.io import savemat
import progressbar

'''
Generates an image set with the same statistics and the light debris data set describred here
(https://doi.org/10.1101/133330s)
'''

##########################
#------CONFIGURATION------
##########################
# Dataset 100.000 Training samples, 10.000 Validation samples, 10.000 Test samples
# Warning: 100.000 images result in 75GB of images during rendering
n_samples = 5000
# if diskspace is limited increase n_batches 
# This will create n batches with n samples each
n_batches = 2

#number of fragments present in each image (10, 30, 50)
n_debris = [10, 11]
#n_debris = [30, 31]
#n_debris = [50, 51]

font_set = ['arial-bold']

for k in range(1, n_batches+1):
    # Generate samples
    clutter_list = [generate.sample_clutter(font_set=font_set) for i in range(n_samples)]

    # Save image set
    clutter_list = io.name_files('A:/digitdebris/' + str(n_debris[0]) + 'debris', clutter_list=clutter_list)
    io.save_image_set(clutter_list, 'A:/digitdebris/' + str(n_samples) + '_' + str(n_debris[0]) + 'debris' + str(k) + '.csv')

    # Render images and save as mat file
    print('Rendering images...')
    sys.stdout.flush()

    bar = progressbar.ProgressBar(maxval=len(clutter_list))
    bar.start()
    for i, cl in enumerate(clutter_list):
        cl.render_occlusion()
        bar.update(i+1)
    print('Saving mat file...')
    fname_list = [cl.fname for cl in clutter_list]
    images_dict = io.save_images_as_mat(abspath('A:/digitdebris/' + str(n_samples) + '_' + str(n_debris[0]) + '_no_debris' + str(k) + '.mat'), 
                                        clutter_list, (32,32), fname_list=fname_list, delete_bmps=True, overwrite_wdir=True)

    # Make debris 
    debris_array = generate.make_debris(n_samples, n_debris=n_debris, font_set=font_set)
    images_with_debris = generate.add_debris(images_dict['images'], debris_array)

    images_with_debris_dict = {
        'images':images_with_debris,
        'targets':images_dict['targets'],
        'binary_targets':images_dict['binary_targets']
        }
    savemat('A:/digitdebris/' + str(n_samples) + '_' + str(n_debris[0]) + 'debris' + str(k) + '.mat', images_with_debris_dict)
    print('Done. Images saved at {0}'.format(abspath('A:/digitdebris/' + str(n_samples) + '_' + str(n_debris[0]) + 'debris' + str(k) + '.mat')))
