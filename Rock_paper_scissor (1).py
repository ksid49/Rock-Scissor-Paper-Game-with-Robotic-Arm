#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob
import csv
from xlsxwriter.workbook import Workbook

print(os.listdir("./inputs1"))


# In[2]:


FAST_RUN = False
IMAGE_WIDTH=648
IMAGE_HEIGHT=488
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_NEW_SIZE=(128,128)
IMAGE_CHANNELS=3


# In[3]:


filenames = os.listdir("./inputs1/train")

def getint(name):
    basename = name.split('.')
    return basename[0],int(basename[1])
filenames.sort(key=getint)

categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'rock':
        categories.append(0)
    elif category == 'paper':
        categories.append(1)
    else:
        categories.append(2)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.head()
#df.tail()
#df['category'].value_counts().plot.bar()


# In[4]:


sample = random.choice(filenames)
image = load_img("./inputs/train/"+sample)
plt.imshow(image)


# In[5]:


#defining classification model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[6]:


earlystop = EarlyStopping(patience=20)
mcp_save = ModelCheckpoint('./inputs1/model.h5', save_best_only=True, monitor='val_loss', mode='min')


# In[7]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[8]:


callbacks = [earlystop, mcp_save, learning_rate_reduction]


# In[9]:


df["category"] = df["category"].replace({0: 'rock', 1: 'paper', 2: 'scissor'}) 
#print(df["category"])


# In[10]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[11]:


train_df['category'].value_counts().plot.bar()


# In[12]:


validate_df['category'].value_counts().plot.bar()


# In[13]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=100


# In[14]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./inputs1/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_NEW_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[15]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./inputs1/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_NEW_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[16]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "./inputs1/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_NEW_SIZE,
    class_mode='categorical'
)


# In[17]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[18]:


epochs=3 if FAST_RUN else 1000
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[19]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[20]:


test_filenames = os.listdir("./inputs1/test")
def getint(name):
    basename = name.split('.')
    return int(basename[0])
test_filenames.sort(key=getint)

test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_df.head()


# In[21]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = validation_generator
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "./inputs1/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_NEW_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[22]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
print(predict)
test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({ 'rock': 0, 'paper': 1, 'scissor': 2})

test_df['category'].value_counts().plot.bar()


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('inputs1/submission_RPS.csv', index=False)

for csvfile in glob.glob(os.path.join('./', '*.csv')):
    workbook = Workbook(csvfile[:-4] + '.xlsx')
    worksheet = workbook.add_worksheet()
    with open(csvfile, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            for c, col in enumerate(row):
                worksheet.write(r, c, col)
    workbook.close()



# In[ ]:




