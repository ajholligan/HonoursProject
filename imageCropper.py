from PIL import Image
import os.path, sys

path = "C:\\Users\\aholl\\OneDrive\\Documents\\Honours Project\\Code\\pytorch-prototypeDL-master\\data\\HAM10000_images_part_1"
path2 = "C:\\Users\\aholl\\OneDrive\\Documents\\Honours Project\\Code\\pytorch-prototypeDL-master\\data"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((75, 0, 525, 450)) #corrected
            filename = f.rsplit('\\', 1)
            imCrop.save(path2 + '\\crop\\' + filename[1] + '.jpg', "JPEG", quality=100)

crop()