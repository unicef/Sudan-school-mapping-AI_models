# UNICEF_Talent

## Content
1. [Project directories structure](#project-directories-structure)
2. [Model files](#model-files)
    * 2.1 [Tensorflow models](#tensorflow-models)
    * 2.2 [YOLTv5](#yoltv5)
      * 2.2.1 [How it works](#how-it-works)
      * 2.2.2 [How to use](#how-to-use)
3. [Pipeline](#pipeline)
    * 3.1 [How pipeline works](#how-pipeline-works)

## Project directories structure

```
data
└─── models                                  - saved models and weights
└─── temp                                    - temporary folder containing images slices, created only during debugging
src                                          - contains the entire source code
└─── stages                                  - pipeline stages scripts
│   └─── preprocessing.py                    - preprocessing tasks
│   └─── prediction.py                       - prediction running tasks
│   └─── postprocessing.py                   - postprocessing tasks
└─── tf                                      - source code related to TensorFlow solution    
│   └─── models                              - TensorFlow models 
│   │   └─── common.py                       - base class implementation inherited by all TensorFlow models
│   │   └─── ensemble.py                     - an ensemble model implementation that combines results from object classifier and object localizer
│   │   └─── object_localizer.py             - object localization model
│   │   └─── object_classifier.py            - object classification model
│   │   └─── loss_learning_rate_scheduler.py - adaptive learning rate scheduler
│   │   └─── loss_utils.py                   - custom loss functions
│   │   └─── model_utils.py                  - utilities for working with models
│   │   └─── patch_encoder.py                - patches encoder for the ViT model
│   │   └─── patches.py                      - creates patches for the ViT model
│   │   └─── resnet.py                       - common functionalities inherited by all ResNet models
│   │   └─── resnet101.py                    - 101-layer ResNet model
│   │   └─── resnet18.py                     - custom 18-layers ResNet model
│   │   └─── resnet50.py                     - 50-layers ResNet model
│   │   └─── run_ensemble.py                 - runs the ensemble model prediction
│   │   └─── transformer.py                  - ViT model implementation
│   │   └─── vgg19.py                        - VGG19 model implementation
│   └─── utils                               - utility code
└─── yolo                                    - source code related to Yolo solution 
│    └─── models                             - Yolo models
└─── balanced_data_generator.py              - balanced data generator implementation
└─── bbox_utils.py                           - utility function used for processing the bounding boxes data
└─── cli.py                                  - provides access to CLI parameters
└─── constants.py                            - constants values shared across the project code
└─── geo_utils.py                            - utility functions for processing the gejson data
└─── img_utils.py                            - utility functions for images processing and dataset creation  
└─── pipeline.py                             - the main pipeline script
└─── utils.py                                - general utility functions
README.md                                    - project description and other documentation
```

## Model files

### TensorFlow models

All TensorFlow models inherit from the BaseModel class stored in the **src/tensorflow/models/common.py** script.\
Prediction with TensorFlow models is run in two phases:
1. Object classification;
2. Object localization.

In the object classification phase an image is split into a number of tiles of specific size and than object classification
is run on each tile that contains schools.\
In the object localization phase each tile from the object classification phase is further split into smaller tiles 
and the object localization is run on each tile. 

The number of tiles is determined using the following formula:
```
tiles_x = math.ceil((image_width - model_input_shape_x) / stride_per_x) + 1
tiles_y = math.ceil((image_height - model_input_shape_y) / stride_per_y) + 1
```
where:
* **image_width, image_height** - dimensions of images used for prediction
* **model_input_shape_x, model_input_shape_y** - dimensions of the images that are used for the model training
* **stride_per_x, stride_per_y** - horizontal and vertical distance between centers of the two adjacent tiles

Two main classes for loading saved models are **ObjectClassifier** (src/tf/models/object_classifier.py) and 
**ObjectLocalizer** (src/tf/models/object_localizer.py).\
The main difference between the object classifier and the object localizer ML models is in the input shape of images that they were trained on.\
The object classifier models are trained on 256x256 images while the object localizer models are trained on smaller image patches like 92x92 pixels in size.

The object classifer and the object localizer use sliding window approach except that the object localizer has additional additional step and that is it selects only those bounding boxes, from the list of overlapping bounding boxes, that have the highest probability that they contain school buildings.

There is additional **EnsembleModel** class that combines output of the object classifier and the object localizer for the final prediction.
<br/>

### YOLTv5

#### How it works
Given a folder of geotiff files or png image files, it uses YOLOv5 model to run inference on geotiff files with image resolution larger than 600x600, perform post-processing and output geojson files that contain locations of schools detected.

#### How to use

#### Installation
	cd src/yolo/YOLTv5/yoltv5/yolov5
	pip install -r requirements.txt

	# update with geo packages
	conda install -c conda-forge gdal
	conda install -c conda-forge osmnx=0.12 
	conda install  -c conda-forge scikit-image
	conda install  -c conda-forge statsmodels
	pip install torchsummary
	pip install utm
	pip install numba
	pip install jinja2==2.10

#### Training
- For training, use **src/yolo/training/train_yolov5_notebook.ipynb**

#### Prediction
- For prediction, go to ```src/yolo/YOLTv5```
- Modify config file```configs/school_detection.yaml```
- Modify the following variables in config file: **weights_file, yoltv5_path, test_im_dir, src_geojson_dir, project_folder**

From YOLTv5 dir, execute the following code:

	cd yoltv5
	./test.sh ../configs/school_detection.yaml
	
- Output geojson files can be found in```YOLTv5/exp/batch_1/results/geojsons_geo_0p6```

#### Post processing
- For processing output files from prediction pipeline, use **src/yolo/post_process/post_process_notebook.ipynb**

Link to original repository - [YOLTv5](https://github.com/avanetten/yoltv5) 

### Pipeline

#### How pipeline works

The main script for the pipeline is the **src/pipeline.py** script.
The script reads parameters required for running the pipeline from the file **params.yaml**.
The structure of the **params.yaml** file is as follows:
```yaml
models:
  MODEL NAME:
    run: COMMAND THAT IS EXECUTED WHEN RUNNING THE MODEL
    params:
       input: THE PATH TO A FOLDER OR A FILE CONTAINING REQUIRED INPUT FOR RUNNING THE MODEL.
       output: THE PATH TO THE FOLDER WITH FILES CONTAINING OUTPUT FROM THE MODEL.
stages:
  STAGE NAME:
    run: COMMAND TO EXECUTE WHEN RUNNING THE STAGE IN THE PIPELINE
    params:
      PARAM 1: PARAM VALUE
      ...
      PARAM N: PARAM VALUE
```
> The **models** property can have more than one model defined.\
> Currently only **tf** (TensorFlow) and **yolo** (YOLO) model names are supported.
> There can be multiple stages in the pipeline as well as multiple parameters per stage.
> Parameters defined for specific model will take precedence over the same parameters defined
> in the prediction stage.

The pipeline script supports following parameter:
* -m: the name of model that will be used for running the predictions, corresponds to model names from the **params**.yaml file. 
* -i: the path to the folder or to the file that is provided as in input for the pipeline stage.
* -o: the path to the folder containing output from a pipeline stage.
* -n: the name of the pipeline stage

> By default the pipeline will run all the stages from the **params.yaml** file in the order 
> how they are defined in the **stages** section.\
> To run just a single stage, use the additional **"-n STAGE_NAME"** parameter with the command 
> that runs the pipeline.

To define a new stage in the pipeline create a new Python script with a mandatory **main** function
and place the script in the **src.stages** folder.
After that create a new stage item in the **params.yaml** file under the **stages** section and add 
required values for the **run** and **params** items.

All path values in the **params.yaml** are relative to the project root folder.\
The pipeline can be started from the command line by positioning to the project root folder and executing 
the following command from the command line:
```bash
python src/pipeline.py
```

By default, the pipeline will run predictions using the YOLO model.\
This can be overridden by providing a value for the "-m" parameter:
```
python src/pipeline.py -m tf
```
The path for the stage input and output can also be provided via **"-i"** option for input and 
**"-o"** option for output path:
```
python src/pipeline.py -m tf -i NEW_INPUT_DIR_PATH -o NEW_OUTPUT_DIR_PATH
```
> Providing path to input or output folders will override specific path provided in the 
> **params.yaml** file.

> If you get the error **ModuleNotFoundError: No module named 'src'** when running the code,
> try adding "." to the PYTHONPATH environment variable.




