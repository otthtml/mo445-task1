import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil


# parameters for the subimages
WIDTH = 64
HEIGHT = 128
Dy = 32
Dx = 32
SIZE = (WIDTH, HEIGHT)
STRIDE = (Dy, Dx)
# THRESHOLD = 0.35

def getImagesPaths(file_path: str):
    with open(file_path, 'r') as fh:
        names = []
        line_index = 0
        for line in fh:
            if line_index >= int(os.getenv('NUMIMAGES')):
                break
            names.append(line.strip())
            line_index += 1
        return names

def readAndGrey(path_to_image: str):
    # read the image, convert it to greyscale
    # colored_image = cv2.imread(path_to_image)
    # grey_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    return grey_image
    # return colored_image

def createSubImage(image: np.ndarray, mask: np.ndarray):
    # find the location of the plate according to the mask
    best_sub_image = np.array(SIZE)
    best_sub_image_ratio = 0
    for y in range(0, image.shape[1], Dy):
        for x in range(0, image.shape[0], Dx):
            try:
                # get a portion of the mask to compare it
                sub_image = mask[ x: x+WIDTH, y: y+HEIGHT ]
                
                # we get the number of plate pixels inside the sub_image
                plate_pixels =  np.sum(sub_image == 255)

                # we get the number of pixels inside the sub_image (usually 3000, given width 100 and height 30)
                total_pixels = sub_image.shape[0] * sub_image.shape[1]

                ratio = plate_pixels/total_pixels
                # if the number of white pixels is great enough...
                # we find it in the original image and return it
                if ratio > best_sub_image_ratio and sub_image.shape == SIZE:
                    best_sub_image = image[ x: x+WIDTH, y: y+HEIGHT ]
                    best_sub_image_ratio = ratio
                    # return sub_image
                
                
            except:
                pass
    return best_sub_image


# create filter that'll slide through image

# slide through image

if __name__ == "__main__":

    # get the first x images (according to the NUMIMAGES enviroment variable)
    imagesPaths = getImagesPaths('./train1.txt')
    masksPaths = [ i.replace('orig', 'mask') for i in imagesPaths ]

    # clean the output
    if os.path.exists('./output/patches'):
        shutil.rmtree('./output/patches', ignore_errors=True)
    if not os.path.exists('./output/patches'):
        os.makedirs('./output/patches')


    for img, msk in zip(imagesPaths, masksPaths):
        # each image is actually a ndarray
        name = img.split('/')[2].split('.')[0]
        image = readAndGrey(img)
        mask = readAndGrey(msk)

        sub_image = createSubImage(image, mask)
        if type(sub_image) == str:
            print('sub_image should be a ndarray. No plate was found! Reduce THRESHOLD')
            continue

        plt.figure()
        plt.imshow(sub_image)
        plt.savefig('./output/patches/'+ name + '.png')
        plt.close()
        # break

        # for si in range(len(sub_images)):
        #     plt.figure()
        #     plt.imshow(sub_images[si])
        #     plt.savefig('./output/patches/'+ name + "_" + str(si) + '.png')
        #     plt.close()
        #     # break
        # break
