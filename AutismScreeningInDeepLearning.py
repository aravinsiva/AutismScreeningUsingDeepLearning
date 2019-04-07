#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Screening and machine learning

import sys
import pandas as pd
import sklearn
import keras




# In[36]:


#Begin building model for training

from keras.models import Sequential
from keras.layers import Dense #Equivalent of a fully connected layer
from keras.optimizers import Adam
from sklearn import model_selection


#define a function to create the Keras Model 

def create_model():
    model=Sequential()
    model.add(Dense(8, input_dim=96, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
    
    #Compile the model
    
    adam=Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model


def generate_model(X,Y):

    #Generate a model
    model= create_model()
    print(model.summary())


    #split the x y data into training anf testing datasets

    X_train, X_test, Y_train, Y_test= model_selection.train_test_split(
                                    X,Y,test_size=0.2)

    # In[38]:


    # fit the model to the training data
    model.fit(X_train,Y_train, epochs= 50, batch_size=10, verbose=1)
    print (X_test.iloc[-1])
    prediction= model.predict_classes(X)
    print prediction
    #prediction= model.predict_classes(test_value)
    return prediction[-1]

def data_preprocess():

    # In[5]:


    #import the dataset
    file= 'Autism-Child-Data.txt'

    data= pd.read_table(file,sep=',',index_col=None)


    # In[9]:


    #print the shape of the dataframe
    print (data.shape)

    #Print the first data point to get an idea of how the data looks
    print(data.iloc[0])


    # In[10]:


    #print out multilple patiets
    data.loc[:10]


    # In[11]:


    #As we can see there is some missing data that we need to account for
    data.describe()


    # In[12]:


    #Find the data types that exist in the dataset
    data.dtypes


    # In[14]:


    #Have to take care of value that are missingf
    #Values that are uneccessary
    #Values that have the wrong object and need to be cast to a different object


    data= data.drop(['result','age_desc'], axis=1)


    # In[15]:


    data.loc[:10]


    # In[16]:


    #create x and y datasets for training
    x= data.drop(['Class/ASD'],1)
    y= data['Class/ASD']


    # In[17]:


    x.loc[:10]
    #As you can see below there are no values for the class


    # In[19]:


    X= pd.get_dummies(x)
    X.loc[:10]
    #Make data categorical


    # In[20]:


    X.columns.values
    #All of the different possible values in the dataset are listed below


    # In[22]:


    #print an example of a patient
    X.loc[0]


    # In[25]:


    # perform the same opperation for the y
    Y= pd.get_dummies(y)
    Y.iloc[:10]

    return [X,Y]


    #End of data preprocessing 






