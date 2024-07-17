import os
import cv2
import numpy as np

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})

for file in os.listdir('img/'):
    img = cv2.imread('img/'+file,0)
    if img.shape != [28,28]:
        img2 = cv2.resize(img,(28,28))
            
    img = img2.reshape(28,28,-1);

    #revert the image,and normalize it to 0-1 range
    img = img/255.0

    with open('pre-proc-img/'+file[:-4]+'.txt','w') as f:
        for i in range(28):
            for j in range(28):
                f.write(str(img[i][j][0])+' ')
            f.write('\n')