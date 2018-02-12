import numpy as np
import cv2
from random import *

num_of_images = 60000
image_size = 224
num_of_channels = 1

min_circle_size = 5
max_circle_size = 25

coord_file = open('centre_coord.txt','w')
for i in range(num_of_images):
	
	radius = randint(min_circle_size,max_circle_size)
	#print(radius)
	
	x_coord = randint(1,image_size)
	while(True):
		if x_coord-radius >= 0 and  x_coord+radius <= image_size:
			break
		else:
			x_coord = randint(1,image_size)
	#print(x_coord)
	
	y_coord = randint(1,image_size)
	while(True):
		if y_coord-radius >= 0 and y_coord+radius <= image_size:
			break
		else:
			y_coord = randint(1,image_size)
	#print(y_coord)

	blank_image = np.zeros((image_size,image_size,num_of_channels), np.uint8)
	cv2.circle(blank_image,(x_coord,y_coord), radius, (255), -1)
	cv2.imwrite('img'+'{:09d}'.format(i+1)+'.png',blank_image)
	coord_file.write(str(x_coord)+','+str(y_coord)+','+str(radius)+'\n')

coord_file.close()