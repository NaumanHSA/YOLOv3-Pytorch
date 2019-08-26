import glob, os
import config

# ============================== Config ========================================

DATA_PATH 				=  config.PARAMS['annotationDirectory']
DATA_PATH_TOWRITE       = 'custom/dataset/'			# Directory where the data will reside, relative to 'darknet.exe'
TEST_PERCENTAGE 		= 20				# Test Image Percentage
IMAGE_FORMAT    		= config.PARAMS['imageFormat']				# Image Format of your data [jpg / png / jpeg]

# ============================== Config ========================================

def process():
    total_images = 0
    file_train = open('train.txt', 'w')
    file_test = open('val.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / TEST_PERCENTAGE)

    for format in IMAGE_FORMAT:
        for pathAndFilename in glob.iglob(os.path.join(DATA_PATH, str("*." + format))):

            basename = os.path.basename(pathAndFilename)

            if counter == index_test:
                counter = 1
                file_test.write(DATA_PATH_TOWRITE + basename + "\n")
            else:
                file_train.write(DATA_PATH_TOWRITE + basename + "\n")
                counter = counter + 1
            total_images += 1

    print('All images Present : ' + str(total_images))
    return


def check_files():

	# Read train.txt and test.txt
	file_train = open('train.txt', 'r')
	file_test = open('val.txt', 'r')

	lines_train = file_train.read().split("\n")
	lines_test = file_test.read().split("\n")

	print('Train images : %s' % (len(lines_train) - 1))
	print('Test images : %s' % (len(lines_test) - 1))

if __name__ == "__main__":

	process()
	check_files()
