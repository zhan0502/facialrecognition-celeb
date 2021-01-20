#remove _MASOX and .DS_store file
import shutil
import os
dest =  '/home/projects/52000146/ACV_P1/testset/testset/' 
files = os.listdir(dest)
for f in files:
    subfolder = dest + f +'/'
    imgs = os.listdir(subfolder)
    for i in range(len(imgs)):
        shutil.move(subfolder + imgs[i], dest)
        if len(os.listdir(subfolder)) == 0: # Check is empty..
            shutil.rmtree(subfolder)