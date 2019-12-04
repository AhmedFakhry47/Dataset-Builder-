from tqdm import tqdm
import numpy as np
import os , sys
import zipfile
import random
import json
import wget
import cv2
import gc

def switcher (str='coco'):
    switch={
    'coco' :['http://images.cocodataset.org/zips/train2017.zip','http://images.cocodataset.org/zips/val2017.zip','http://images.cocodataset.org/annotations/annotations_trainval2017.zip','http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'],
    'def': ['http://images.cocodataset.org/zips/train2017.zip','http://images.cocodataset.org/zips/val2017.zip','http://images.cocodataset.org/annotations/annotations_trainval2017.zip','http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip']
    }
    try :
        return switch['coco']
    except KeyError:
        print('Right now , only coco dataset is available')
        exit()

def visualize_img(img,bboxes,color,thickness,classes):
    bboxes_points=[]
    for b in bboxes:
        startpoint = (int(b[0]),int(b[1]))
        endpoint = (int(b[0]+b[2]),int(b[1]+b[3]))
        bboxes_points.append([startpoint,endpoint])
    for i,box in enumerate(bboxes_points):
        box = cv2.rectangle(img,box[0],box[1],color,thickness)
    cv2_imshow(box)
    return 

class imobj :
    '''
    Image objects for lazy evaluation
    that's a different approach from the one used in other builders
    '''
    def __init__(self,path):
        self.path = path
    def eval (self):
        return(cv2.imread(self.path))


class gp_builder:
    def __init__(self,str='coco'):
        self.urls = switcher(str)
        self.data_files = []
        self.train_dataset = []
        self.val_dataset=[]

    def download_and_prepare(self):

        #downlaod data
        for url in tqdm(self.urls):
            wget.download(url)
            self.data_files.append(url.split('/')[4])

        #unzipping data files
        for i,data_file in enumerate(self.data_files):
            with zipfile.ZipFile('/content/'+data_file, 'r') as zipped:
                zipped.extractall('/content')

            #to remove .zip extension
            if(i==2 ):
                self.data_files[i]= '/content/annotations'
                continue
            elif (i==3):
                self.data_files[i]='/content/sample_data'
                continue
            else :
                self.data_files[i] = '/content/'+data_file.split('.')[0]

        self.prepare()

    def get_andata(self):
        print(self.data_files)
        annotations_dirs = [self.data_files[2]+'/instances_train2017.json',self.data_files[2]+'/instances_val2017.json']
        train_andata = dict()
        val_andata = dict()

        for itr , annotation_dir in enumerate(annotations_dirs):
            with open(annotation_dir,'r') as json_file:
                if (itr==0):train_andata=json.load(json_file)
                else: val_andata = json.load(json_file)
        return train_andata, val_andata

    #A function that returns the label ( class / category )of the current bounding box
    def get_cat (self,id,andata):
        return next(item["name"] for item in andata["categories"] if item["id"]==id)

    def get_bbox(self,train_andata,val_andata):
        timg_bbox = dict()
        vimg_bbox = dict()

        for itr,andata in enumerate([train_andata,val_andata]):
            img_bbox = dict()
            for annotation in andata['annotations']:
                current_id = annotation["image_id"]
                if current_id in img_bbox.keys():
                    img_bbox[current_id].append([annotation["bbox"],self.get_cat(annotation["category_id"],andata)])
                else :
                    img_bbox[current_id] = [[annotation["bbox"],self.get_cat(annotation["category_id"],andata)]]

            if (itr==0):timg_bbox = img_bbox.copy()
            else: vimg_bbox = img_bbox.copy()
        return timg_bbox, vimg_bbox

    def get_key(self,string):

        '''
        this function returns the image id from image file stored in the disk
        if image file is stored as ' 000546.png'
        image ID will be 546
        and that what hopefully this function does.
        '''
        string =string.split('.')[0]
        for i,j in enumerate(string):
            if(j!='0'):
                break
        return int(string[i:])

    def get_imgs(self):
        imgs_dir = [self.data_files[0],self.data_files[1]]
        timgs=dict()
        vimgs=dict()
        corrupted_keys=[]

        for itr,direc in enumerate(imgs_dir) :
            imdict=dict()
            all_images_names = [img for img in os.listdir(direc) if os.path.isfile(os.path.join(direc,img))]

            for image in tqdm(all_images_names):
                key = self.get_key(image)
                try:
                    im = imobj(os.path.join(direc,image))
                    imdict[key]=im
                except :
                    corrupted_keys.append(key)
                    continue

            if (itr==0): timgs = imdict
            else : vimgs = imdict

        return timgs,vimgs

    def prepare(self):
        train_andata,val_andata = self.get_andata()
        timg_bbox ,vimg_bbox = self.get_bbox(train_andata,val_andata)
        timgs , vimgs =self.get_imgs()

        for key in sorted(timgs.keys()):
            try:
                self.train_dataset.append([timgs[key],timg_bbox[key]])
            except :continue

        for key in sorted(vimgs.keys()):
            try:
                self.val_dataset.append([vimgs[key],vimg_bbox[key]])
            except : continue

        random.shuffle(self.train_dataset)
        random.shuffle(self.val_dataset)

        self.report(timgs,timg_bbox)

    def report(self,timgs,timg_bbox):
        first = sorted(timg_bbox.keys())
        second= sorted(timgs.keys())
        found =0
        not_found=0
        for keyi in second:
          try :
            opn = timg_bbox[keyi]
            opn2= timgs[keyi]
            found+=1
          except :
            not_found +=1
        print('Succeful engagement : {} , Unsecceful engagement : {}'.format(found,not_found))
        print('length of read imags : {} , length of annotations corresponding to images {}'.format(len(first),len(second)))
        print('length of downloaded images : {}'.format(len(os.listdir('/content/train2017'))))

    def _get_data(self):
        return self.train_dataset,self.val_dataset

    def _get_traindata(self):
        return self.train_dataset

    def _get_valdata(self):
        return self.val_dataset
