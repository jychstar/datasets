Street View House Number (SVHN) dataset is hosted at http://ufldl.stanford.edu/housenumbers/

- 99.3 k instances (73.3 k training, 26 k testing, 531 k bonus)
- input features depend on formats
- 1 output feature with 10 factors, note that 10 is for 0. (The reason doing so is simply the creaters user matlab and matlab begins with 1.)

## format 1

take train.tar.gz for example:

1. after unzipping, get a "train" folder with 33.402 k png files and 1 digitStruct.mat file, totaling 626 MB. 

2. Use matlab open the .mat file, get 1*33402 struct. Each struct has 2 fields: name and bbox. name is for png file. bbox is the pixel position of the blue box, which is represent by a struct with 5 fields: height, left, top, width, label. 

3. Note that the bboxes don't have a fixed size. So we have to use cv2 to resize it. The bunch of codes is in the accompany ipynb file.

4. Machine learning data is usually in the 2D-shape (num, featuers), but why image data is typical stored in 3D-shape: (num, n_x, n_y)  or even 4D-shape (num,n_x,n_y,3) ? 1st, it is easier to extract individual pixel or a rectangle block of pixels. 2nd, it is natural to show the image by `matplotlib.pyplot.imshow(image)`

### HDF5

http://docs.h5py.org/en/latest/quick.html
HDF stands for Hierarchical Data Format. 

An HDF5 file is a container for two kinds of objects:

1. **datasets**. array-like collection of data. work like numpy arrays.
2. **groups**. folder-like containers that hold dataset and other group. work like dictionaries. 

## format 2

The images are 90% preprocessed. The mat file in format 1 is HDF5 format, but the mat file in format 2 is not. So `scipy.io.loadmat` can easily handle the work.  Besides the format, the major difference is actually the content. In format 2, the image are strictly cropped by 32\*32 pixels. As a result, the adjecent numbers become the background noise to the target number. So you expect the identification accuracy may decrease. 

# digit_recognition  project

[project](https://github.com/udacity/machine-learning/tree/master/projects/digit_recognition) and [rubic](https://review.udacity.com/#!/rubrics/413/view)

   