## Data Preparation

### Preparation for C20K dataset

- Download the COCO 2014 dataset and put it in `data/`. 
- Download `mixed_lite.mat` and `mixed_image_paths.mat` for C20K dataset [here](https://drive.google.com/drive/folders/16NT4KD3HIF8rTHYSKqtq3SqZEFlv1s5h?usp=sharing) and put it in `data/coco_train_20k/mixed` then run the following code in Matlab **from the LOD folder** to modify the image paths according to your LOD path.
```
setup;
imdb = load(fullfile(LOD_ROOT, 'data/coco_train_20k/mixed/mixed_image_paths_original.mat'));
imdb.image_paths = cellfun(@(el) fullfile(LOD_ROOT, 'data', el), imdb.image_paths, 'Uni', false);
savefile(fullfile(LOD_ROOT, 'data/coco_train_20k/mixed/mixed_image_paths.mat'), imdb);
```

### Use your own dataset

LOD requires the data to be organized as follow:
```
data/
|   dataset1/
|   |   class1/
|   |   |   class1_lite.mat
|   |   |   class1_image_paths.mat
|   |   class2/
|   |   |   class2_lite.mat
|   |   |   class2_image_paths.mat
|   |    ...
|   dataset2/
|    ...

```

Through the project, we suppose that all datasets have a single class `mixed`. Its information is contained in two `.mat` files, `mixed_lite.mat` and `mixed_image_paths.mat`. `mixed_lite.mat` is a struct with the following fields:
```
bboxes: (n x 1) cell, each cell is a (K x 4) matrix containing ground-truth bounding boxes of the images in the dataset.
        The bounding boxes are in format [x1,y1,x2,y2], inclusive.
        THIS FIELD IS ONLY USED FOR EVALUATION. YOU DO NOT NEED IT TO RUN THE METHOD.
        IN THE CASE GROUND TRUTH IS NOT AVAILABLE, YOU CAN SET bboxes=cell(n,1)
images_size: (n x 1) cell, containing images' size (height x width).
``` 
and `mixed_image_paths.mat` is a struct with a single field
```
image_paths: (n x 1) cell, containing paths to the images in the dataset.
```

To use your own dataset, create a `mixed_lite.mat` and a `mixed_image_paths.mat` with the fields above and put it in the right place. See the `.mat` files in `data/coco_train_20k/mixed` for an example.

