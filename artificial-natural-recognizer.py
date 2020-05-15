
# coding: utf-8

# In[1]:


import numpy as np
import keras


# In[2]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten


# In[3]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True)


# In[4]:


x_train=train_datagen.flow_from_directory(r'C:\Users\Nikhil\Desktop\cnn-project\imagesets\imagesets',target_size=(64,64),batch_size=30,class_mode='binary')


# In[5]:


x_train.class_indices


# In[6]:


cnn_model=Sequential()


# In[7]:


cnn_model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# In[8]:


cnn_model.add(Flatten())


# In[9]:


cnn_model.add(Dense(150,activation='relu'))


# In[10]:


cnn_model.add(Dense(1,activation='sigmoid'))


# In[11]:


cnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[12]:


cnn_model.fit_generator(x_train,samples_per_epoch=1000,epochs=7,nb_val_samples=200)


# In[13]:


cnn_model.save('artificial-natural.h5')


# In[14]:


from keras.models import load_model
import numpy as np
import cv2
model=load_model('artificial-natural.h5')


# In[15]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[16]:


from skimage.transform import resize
def detect(frame):
    try:
        img=resize(frame,(64,64))
        img=np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img=img/255.0
        prediction=model.predict(img)
        print(prediction)
        prediction_class=model.predict_classes(img)
        print(prediction_class)
    except AttributeError:
        print('shape not found')


# In[17]:


frame=cv2.imread(r"C:\Users\Nikhil\Desktop\cnn-project\imagesets\test\butterfly-1127666__340.jpg")
data=detect(frame)


# In[18]:


import numpy as np
from keras.preprocessing import image

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras.models import load_model
classifier = load_model('artificial-natural.h5')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index=["artifical","natural"]
    label = Label( root, text="Prediction : "+index[result[0][0]])
    label.pack()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()

btn = Button(root, text='open image', command=open_img).pack()

root.mainloop()

