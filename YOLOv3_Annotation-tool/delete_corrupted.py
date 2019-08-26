import glob, os
from PIL import Image
import config

# ============================== Config ========================================

DATA_PATH = config.PARAMS['imageDirectory']
TEST_PERCENTAGE = 15  # Test Image Percentage
IMAGE_FORMAT = config.PARAMS['imageFormat']  # Image Format of your data [jpg / png / jpeg]


# ============================== Config ========================================

def delete_currupted():

    currupted_Images = []

    for format in IMAGE_FORMAT:
        for pathAndFilename in glob.iglob(os.path.join(DATA_PATH, str("*." + format))):
            basename = os.path.basename(pathAndFilename)

            try:
                img = Image.open(pathAndFilename)  # open the image file
                img.verify()  # verify that it is, in fact an image

            except:
                currupted_Images.append(basename.split('.')[0])
                os.remove(pathAndFilename)
                try:
                    os.remove(pathAndFilename.replace(format, 'txt'))
                except:
                    continue

    print('Currupted images deleted')
    for image in currupted_Images:
        print(image)
    print('\nTotal Currupted Images : ' + str(len(currupted_Images)))


if __name__ == "__main__":

    delete_currupted()
