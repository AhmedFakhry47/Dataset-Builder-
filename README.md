# Dataset-Builder-
A class that is used to download and prepare data set ( coco dataset and other datasets ) . It's similar to tfds builder but faster.

## Dependencies you will need to run the code 
-tqdm <br/>
-numpy <br/>
-os <br/>
-sys <br/>
-zipfile <br/>
-random <br/>
-json <br/>
-wget <br/>
-cv2 <br/>
-gc <br/>


## Steps of work 
After importing the file , Create an instance of gp_builder class . Then you have two methods in order to download and prepare data 
First method is : download_and_prepare() 
Second method is : _get_data()

like in the following picture : 
![1](https://user-images.githubusercontent.com/44531149/70165427-8db76500-16cb-11ea-8dd3-3505396c7b6c.png)


Expected output would be :
![2](https://user-images.githubusercontent.com/44531149/70165423-8c863800-16cb-11ea-8dd7-ed55a94a9171.png)

In data_builder.py , there is a function that is used to visualize images from the dataset <br/>
and the expected output would be like :  <br/>
![3](https://user-images.githubusercontent.com/44531149/70165411-898b4780-16cb-11ea-9dc8-6b333c3cc64c.png)




