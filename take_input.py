# input_.py

import numpy as np
import cv2 as cv
import os


def make_val_test(train_csv,val_frac,test_frac):
    train_csv=train_csv.sample(frac=1.0)
    test_size=int(train_csv.shape[0]*test_frac)
    val_size=int(train_csv.shape[0]*val_frac)
    test=train_csv.iloc[:val_size,:]
    val=train_csv.iloc[val_size:test_size+val_size,:]
    train=train_csv.iloc[test_size+val_size:,:]
    
    return train,val,test

def make_batch(data):
    PATH=os.path.join(os.getcwd(),'images')
    i=0
    for x in range(data.shape[0]):
        im=data.iloc[x][0]
        img_addr=os.path.join(PATH,im)
        img=np.reshape(cv.imread(img_addr),[-1,480,640,3])
        labels=np.array([data.iloc[x][1],data.iloc[x][2],data.iloc[x][3],data.iloc[x][4]])
        if i>0:
            images=np.vstack((images,img))
            target=np.vstack((target,labels))
        else:
            images=img
            target=labels
        i+=1
    return images,target


def batch(train,batch_size):
    PATH=os.path.join(os.getcwd(),'images')
    i=0
    while i<train.shape[0]:
        x=0
        while x<batch_size and i<train.shape[0]:
            im=train.iloc[i][0]
            img_addr=os.path.join(PATH,im)
            img=np.reshape(cv.imread(img_addr),[-1,480,640,3])
            #img=cv.cvtColor()
            labels=np.array([train.iloc[i][1],train.iloc[i][2],train.iloc[i][3],train.iloc[i][4]])
            
            if x>0:
                data=np.vstack((data,img))
                target=np.vstack((target,labels))
            else:
                data=img
                target=labels
            
            x+=1
            i+=1
        yield data,target
        

def show(data,target,orig=1):
    for x in range(data.shape[0]):
        img=data[x]
        labels=target[x]
        
        cv.rectangle(img,(labels[0],labels[2]),(labels[1],labels[3]),(0,0,0),5)
        '''if orig.any()!=1:
                                    labels_orig=orig[x]
                                    cv.rectangle(img,(labels_orig[0],labels_orig[2]),(labels_orig[1],labels_orig[3]),(0,255,0),5)'''
        cv.imshow('img',img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
