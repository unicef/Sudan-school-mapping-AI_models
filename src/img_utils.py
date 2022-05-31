import os
import cv2
import math
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Tuple, Callable, Dict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.balanced_data_generator import BalancedDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def pad(src_img, tile_width, tile_height):
    """
    Add black pixels to image borders to make it ready for splitting.
    arguments:
        src_img (PIL image (or) np array): image to be padded
        tile_width (int): tile width
        tile_height (int): tile height

    returns:
        PIL image: padded image
    """

    img_type = type(src_img)

    if img_type == np.ndarray:
        img = src_img.copy()
    else:
        # convert to numpy array
        img = np.array(src_img)

    img_width = img.shape[1]
    img_height = img.shape[0]

    pad_width = int((np.ceil(img_width / tile_width) * tile_width) - img_width)
    pad_height = int((np.ceil(img_height / tile_height) * tile_height) - img_height)

    result_image = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT)

    # convert numpy array to PIL image format
    result_image = Image.fromarray(result_image.astype('uint8'), 'RGB')

    return result_image


def get_categories_weights(categories: Dict[str, List[np.array]], total_images: int) -> Dict[str, float]:
    """
    Calculates weights for each image category.

    :param categories: a dictionary with mapping between different image categories and
    a list of images belonging to a specific category.
    :param total_images: total number of images for all categories.
    :return: a dictionary containing categories as keys and calculated weights as values.
    """
    categories_len = len(categories)
    categories_weight = {}
    for index, (category, images) in enumerate(categories.items()):
        category_weight = total_images / (categories_len * len(images))
        if not category in categories_weight:
            categories_weight.update({category: category_weight})
    return categories_weight


def get_images_per_category(images_dir: str) -> Tuple[Dict[str, List[str]], int]:
    """
    Iterates over subfolders that are found in the specified images folder
    and creates a dictionary with categories (subfolders) and a list of image
    names belonging to each category.

    :param images_dir: absolute path to the parent folder
    :return: a tuple with a dictionary containing mappings of
    subfolder names and image names that are found in each subfolder and
    a total number of images found.
    """
    categories = {}
    total_images = 0
    for category in os.listdir(images_dir):
        category_path = os.path.join(images_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            categories.update({category: images})
            total_images += len(images)
    return categories, total_images


def create_csv(images_dir: str, csv_dest_path=None):
    """
    Creates a DataFrame object containing image names, their categories and calculated weights.
    Image category is determined from the name of a folder where the image is stored in.
    Image weight is calculated per category and it depends on the number of images in a folder,
    the greater the number of images for a category the lower weight for that category.

    :param images_dir: the path to the folder with all images.
    :param csv_dest_path: the path to the CSV file containing information about images.
    :return: None
    """
    categories, total_images = get_images_per_category(images_dir)
    categories_weight = get_categories_weights(categories, total_images)
    data = []

    for index, (category, images) in enumerate(categories.items()):
        for img_name in images:
            data.append([os.path.join(images_dir, category, img_name), category, category_weight])
    df = pd.DataFrame(data=data, columns=["imagepath", "category", "weight"])
    if csv_dest_path:
        df.to_csv(csv_dest_path, index=False)

    return df, categories_weight


def get_image_data_generator(preprocess_fn: Callable, augment_data: bool=False) -> ImageDataGenerator:
    """
    Creates and return an ImageDataGenerator object

    :param preprocess_fn: a preprocessing function that will run after the image transformation is applied
    :param augment_data: should images be augmented
    :return: new instance of the ImageDataGenerator
    """
    args = { "preprocessing_function": preprocess_fn }
    if augment_data:
        args.update(dict(
            rotation_range=15,
            brightness_range=(0.9, 1.1),
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        ))
    data_gen = ImageDataGenerator(**args)

    return data_gen


def get_datasets(class_images: Dict[str, Dict], split_train_val: Tuple[float, float]) -> \
        Tuple[Tuple[np.array, np.array, np.array], Tuple[np.array, np.array, np.array]]:
    """
    Creates training, validation and test datasets from a dictionary containing mappings
    of image categories and images belonging to each category.

    :param class_images: a dictionary of images categories and images belonging to each category
    :param split_train_val: a tuple containing percentage of training and validation data, the sum
    of percentages should be less than 1.
    :return: a tuple containing two tuples with the input and target data.
    """
    X_train_all = []; X_val_all = []; X_test_all = []
    y_train_all = []; y_val_all = []; y_test_all = []
    for _, data in class_images.items():
        class_index = data["index"]
        class_images = data["images"]
        images_len = len(class_images)
        train_size = int(images_len * split_train_val[0])
        val_size = math.ceil(images_len * split_train_val[1])
        test_size = images_len - train_size - val_size
        X_train = class_images[:train_size]
        X_val = class_images[train_size:train_size+val_size]
        X_test = class_images[-test_size:] if test_size > 0 else []
        y_train = [class_index] * train_size
        y_val = [class_index] * val_size
        y_test = [class_index] * test_size
        X_train_all.extend(X_train)
        X_val_all.extend(X_val)
        X_test_all.extend(X_test)
        y_train_all.extend(y_train)
        y_val_all.extend(y_val)
        y_test_all.extend(y_test)
    X_train_all = np.array(X_train_all)
    X_val_all = np.array(X_val_all)
    X_test_all = np.array(X_test_all)
    y_train_all = np.array(y_train_all)
    y_val_all = np.array(y_val_all)
    y_test_all = np.array(y_test_all)

    return (X_train_all, X_val_all, X_test_all), (y_train_all, y_val_all, y_test_all)


def load_class_images(images_dir: str) -> Dict[str, Dict]:
    """
    Loads images and creates a dictionary with mappings between image
    categories(classes) and a list containing binary data of each image
    that belongs to a specific category

    :param images_dir: a path to the images folder
    :return: a dictionary with image categories and lists of images for each category.
    """
    classes = os.listdir(images_dir)
    if len(classes) == 0:
        raise Exception(f"No folders found in {images_dir}, cannot create data generators.")
    class_images = {}
    for class_index, class_name in enumerate(classes):
        images = []
        for entry in os.scandir(os.path.join(images_dir, class_name)):
            img = load_img(entry.path)
            img_arr = img_to_array(img)
            images.append(img_arr)
        images = np.array(images)
        class_images[class_name] = {"index": class_index, "images": images}

    return class_images


def get_balanced_data_generators(images_dir: str, split_train_val: Tuple[float, float]=(0.8, 0.1),
                                 batch_size: int=32, preprocess_fn: Callable=None) \
        -> Tuple[BalancedDataGenerator, BalancedDataGenerator, BalancedDataGenerator]:
    """
    Creates balanced data generators for the training, validation and the test dataset.

    :param images_dir: path to the images folder
    :param split_train_val: a tuple containing values for percentage of training and validation data
    :param batch_size: number of images per single batch
    :param preprocess_fn: a preprocessing function that should be applied during images preprocessing.
    :return: a tuple of balanced training, validation and test datasets.
    """
    class_images = load_class_images(images_dir)
    (X_train, X_val, X_test), (y_train, y_val, y_test) = get_datasets(class_images, split_train_val)
    train_datagen = get_image_data_generator(preprocess_fn, augment_data=True)
    val_datagen = get_image_data_generator(preprocess_fn)
    test_datagen = get_image_data_generator(preprocess_fn)
    balanced_train_datagen = BalancedDataGenerator(X_train, y_train, train_datagen, batch_size=batch_size)
    balanced_val_datagen = BalancedDataGenerator(X_val, y_val, val_datagen, batch_size=batch_size)
    balanced_test_datagen = BalancedDataGenerator(X_test, y_test, test_datagen, batch_size=batch_size)

    return balanced_train_datagen, balanced_val_datagen, balanced_test_datagen