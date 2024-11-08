'''from skimage import data
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))

coffee = data.coffee()
plt.subplot(1,2,1)

plt.imshow(coffee)

gray_coffee = rgb2gray(coffee)
plt.subplot(1,2,2)
plt.imshow(gray_coffee,cmap="gray")

hsv_coffee = rgb2hsv(coffee)
plt.subplot(1,2,2)

hsv_coffee_colorbar = plt.imshow(hsv_coffee)
plt.colorbar(hsv_coffee_colorbar,fraction = 0.046,pad = 0.04)'''

from skimage import data
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import csv
def create():
        f = open('colour.csv','w+', newline = '')
        w = csv.writer(f)
        ch = 'y'
        w.writerow(['Element',' L Hue','Saturation','Value'])
        while ch in 'Yy':
                ele = input('Enter the Element name : ')
                lh = int(input('Enter the lower Hue : '))
                uh = int(input('Enter the upper Hue : '))
                ls = int(input('Ente the lower Saturation : '))
                us = int(input('Enter the upper Saturation : '))
                lv = int(input('Enter the lower Value : '))
                uv = int(input('Enter the upper Value : '))
                #h = [lh,uh]
                #s = [ls,us]
                #v = [lv,uv]
                l = [ele,lh,uh,ls,us,lv,uv]
                w.writerow(l)
                ch = input('do you want to enter more records:')
                if ch not in 'Yy':
                        break
        f.close()
def display ():
        f = open('colour.csv','r')
        r = csv.reader(f)
        d =[]
        for i in r:
                d.append(i)
        print(d)
        f.close()
def search ():
        f = open('colour.csv','r',newline = '')
        r = csv.reader(f)
        ch = 'y'
        while ch in 'Yy':
                a = input('Enter the Element to be searched :')
                for i in r :
                    if i [0] == a :
                             print('Element : ',i[0],'Hue : ',i[1],'Saturation : ',i[2],'Value : ',i[3]) 
                ch = input('do you want to search more records ?:')
                if ch not in 'Yy':
                        break
        f.close()
def add():
        f= open('colour.csv','a',newline = '')
        w = csv.writer(f)
        ch = 'y'
        while ch in 'yY' :
                ele = input('Enter the Element name : ')
                lh = int(input('Enter the lower Hue : '))
                uh = int(input('Enter the upper Hue : '))
                ls = int(input('Ente the lower Saturation'))
                us = int(input('Enter the upper Saturation : '))
                lv = int(input('Enter the lower Value : '))
                uv = int(input('Enter the upper Value : '))
                #h = [lh,uh]
                #s = [ls,us]
                #v = [lv,uv]
                l = [ele,lh,uh,ls,us,lv,uv]
                w.writerow(l)
                 
                ch = input('do you want to enter more records:? ')
                if ch not in 'yY' :
                        break
        f.close()

#C:/Users/reddy/Downloads/IMG-20241009-WA0031.jpg
def grayscale():
	im1 = plt.imread("C:/Users/reddy/Downloads/IMG-20241009-WA0031.jpg")

	r,g,b = im1[:,:,0],im1[:,:,1],im1[:,:,2]

	gamma = 1.04

	rc,gc,bc = 0.2126, 0.7152, 0.0722

	#print("Enter 1 to convert a defalut image to grayscale image")
	#plt.figure(figsize=(10,10))

	#coffee = data.coffee()
	#plt.subplot(1,2,2)

	'''plt.imshow(im1)
	plt.show()'''

	gsi = rc*r**gamma + gc*g**gamma + bc*b**gamma

	fig = plt.figure(1)
	ar1 = fig.add_subplot(121)
	ar2 = fig.add_subplot(122)

	'''plt.imshow(im1)
	plt.show()'''


	ar1.imshow(im1)
	ar2.imshow(gsi, cmap = plt.cm.get_cmap('gray'))
	fig.show()
	plt.show()

	'''gray_coffee = rgb2gray(im1)
	plt.subplot(1,2,2)
	plt.imshow(gray_coffee)
	plt.show()'''

	#hsv_coffee = rgb2hsv(im1)
	#plt.subplot(1,2,2)

	#hsv_coffee_colorbar = plt.imshow(hsv_coffee)
	#plt.colorbar(hsv_coffee_colorbar,fraction = 0.046,pad = 0.04)



#C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184435.png
#C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184535.png
def coll():
	img1 = cv2.imread("C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184435.png")
	img2 = cv2.imread("C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184535.png")
	img3 = cv2.imread("C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184747.png")
	img4 = cv2.imread("C:/Users/reddy/OneDrive/Pictures/Screenshots/Screenshot 2024-10-09 184815.png")

	img1 = cv2.resize(img1,(150,150))
	img2 = cv2.resize(img2,(150,150))
	img3 = cv2.resize(img3,(150,150))
	img4 = cv2.resize(img4,(150,150))
	
	hst1 = np.hstack([img1,img2])
	hst2 = np.hstack([img3,img4])
	
	vst = np.vstack([hst1,hst2])


	cv2.imshow("collage",vst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

	
	
def u_netmod(input_shape=(128,128,3), num_classes=2):
        inputs = layers.Input(shape=input_shape)
	
	#cont path or encoder
        c1 = layers.Conv2D(64,(3,3),activation = 'relu', padding = 'same')(inputs)
        c1 = layers.Conv2D(64,(3,3),activation = 'relu', padding = 'same')(c1)
        p1 = layers.MaxPooling2D((2,2))(c1)
	
        c2 = layers.Conv2D(128,(3,3), activation = 'relu', padding = 'same')(p1)
        c2 = layers.Conv2D(128,(3,3), activation = 'relu' ,padding = 'same')(c2)	
        p2 = layers.MaxPooling2D((2,2))(c2)
	
        c3 = layers.Conv2D(256,(3,3),activation='relu',padding = 'same')(p2)
        c3 = layers.Conv2D(256,(3,3),activation='relu',padding = 'same')(c3)
        p3 = layers.MaxPooling2D((2,2))(c3)
	
	#bt neck
        c4 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(p3)
        c4 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(c4)
	
	#ex path or decoder
        u5 = layers.Conv2DTranspose(256,(2,2), strides = (2,2), padding = 'same')(c4)
        u5 = layers.concatenate([u5,c3])
        c5 = layers.Conv2D(256,(3,3),activation='relu',padding = 'same')(u5)
        c5 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(c5)
	
        u6 = layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
        u6 = layers.concatenate([u6,c2])
        c6 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(u6)
        c6 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(c6)
	
        u7 = layers.Conv2DTranspose(64,(2,2),strides = (2,2),padding='same')(c6)
        u7 = layers.concatenate([u7,c1])
        c7 = layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same')(u7)
        c7 = layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same')(c7)
	
        outputs = layers.Conv2D(num_classes,(1,1),activation  = 'softmax')(c7)
	
        mode1 = models.Model(inputs=[inputs],outputs=[outputs])
	
        return mode1


#compiling and summerizing the model
#model = u_netmod(input_shape=(128,128,3),num_classes=2)
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()
#dummy func for dataset
def gendumdata(num_samples=100,img_size=(128,128,3),num_classes=2):
	images = np.random.rand(num_samples,*img_size)
	masks = np.random.randint(0,num_classes,(num_samples,img_size[0],img_size[1],num_classes))
	return images,masks
	
	
#training data
'''x_train,y_train = gendumdata()'''

#model training
#history = model.fit(x_train,y_train,epochs=10,batch_size = 16,validation_split=0.1)

#training results
def plot_history(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'],loc='upper left')
	plt.show()
	
	
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'],loc = 'upper left')
	plt.show()
	
#plot_history(history)
#C:/Users/reddy/Downloads/Screenshot 2024-10-24 132141.png

def eleidentify() :
	img = cv2.imread("C:/Users/reddy/Downloads/Screenshot 2024-10-24 132141.png")
	

	#cv2.imshow(img)
	img = cv2.resize(img,(600,600))
	clicked = False
	
	def draw_function(event,x,y,flag,param):
		global clicked,r,g,b
		if evevt == cv2.EVENT_LBUTTONDBLCLK:
			clicked = True
			b,g,r = img[y,x]
			b,g,r = int(b),int(g),int(r)

	cv2.namedWindow("collage")
	cv2.setMouseCallback('collage',draw_function)
	cv2.imshow("collage",img)
	
	#def getColor(R,G,B):
		
	
	
	'''cv2.namedWindow('Color detection')
	cv2.setMouseCallback('Color detection',draw_function)
	#cv2.imshow("Color detection",img)'''
	
	
	
	while True:
		cv2.imshow("collage",img)
		if clicked:
			cv2.rectangle(img,(10,20),(600,60),(b,g,r),-1)
			text = getColor(r,g,b)+ ' ('+str(r) +','+str(g) +','+str(b)+')'
	#plt.figure(figsize =(20,8))
	
	
	grid_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	#plt.figure(figsize=(20,8))
	#plt.imshow(grid_rgb)
	grid_hsv = cv2.cvtColor(grid_rgb,cv2.COLOR_RGB2HSV)
	lower = np.array([174,90.9,86.3])
	upper = np.array([174,90.9,86.3])
	mask = cv2.inRange(grid_hsv,lower,upper)
	#plt.figure(figsize=(20,8))
	cv2.imshow('collage',mask)
	res = cv2.bitwise_and(grid_rgb,grid_rgb,mask = mask)
	plt.figure(figsize =(20,8))
	cv2.imshow("collage",res)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
		
	



'''def eleidentify() :
	img = cv2.imread("/home/acer/Downloads/Rahul24/castshow.png")
	
	plt.figure(figsize =(20,8))
	plt.imshow(img)
	
	#grid_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	#plt.figure(figsize=(20,8))
	#plt.imshow(grid_rgb)
	
	grid_hsv = cv2.cvtColor(grid_rgb,cv2.COLOR_RGB2HSV)
	lower = np.array([115,150,50])
	upper = np.array([125,255,255])
	mask = cv2.inRange(grid_hsv,lower,upper)
	plt.figure(figsize=(20,8))
	plt.imshow(mask)
	res = cv2.bitwise_and(grid_rgb,grid_rgb,mask = mask)
	plt.figure(figsize =(20,8))
	plt.imshow(res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()'''
	
	
	
'''print("Enter 1 to train the program ")
print("Enter 2 to convert a picture from colourful to grayscale")
print("Enter 3 to merge multiple small scale image to a large scale image ")
print("Enter 4 to identify the different elements ")

	
	
choice = int(input("Enter the choice : "))
while(choice>0 and choice<5):
        if choice == 1:
                #compiling and summerizing the model
                model = u_netmod(input_shape=(128,128,3),num_classes=2)
                model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
                model.summary()
                x_train,y_train = gendumdata()
                history = model.fit(x_train,y_train,epochs=10,batch_size = 16,validation_split=0.1)
                plot_history(history)


        elif choice == 2:
                grayscale()


        elif choice == 3:
                coll()
                
        elif choice == 4:
        	eleidentify()

        else :
                break

        print("Enter 1 to train the program ")
        print("Enter 2 to convert a picture from colourful to grayscale")
        print("Enter 3 to merge multiple small scale image to a large scale image ")
        print("Enter 4 to identify the different elements : ")

        choice = int(input("Enter the choice : " ))'''


print("Enter 1 to create and enter the HSV values ")
print("Enter 2 to display all element's HSV values")
print("Enter 3 to search HSV value for one element ")
print("Enter 4 to add new element's HSV vlaues without creating new file ")
print("Enter the choice : ")
choice =  int(input("Enter your choice : "))
                   
while(choice >0 and choice< 5):
        if choice == 1:
                create()

        elif choice == 2:
                display()

        elif choice == 3 :
                search()

        elif choice == 4:
                add()

        print("Enter 1 to create and enter the HSV values ")
        print("Enter 2 to display all element's HSV values")
        print("Enter 3 to search HSV value for one element ")
        print("Enter 4 to add new element's HSV vlaues without creating new file ")
        
        
                     
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

