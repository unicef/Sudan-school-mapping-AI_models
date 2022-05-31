import os
import cv2
import math
from pathlib import Path


SCRIPT_PATH = os.path.join(os.path.dirname(__file__))
PREDICTION_RESULTS_DIR = os.path.join(SCRIPT_PATH, "..", "..", "output", "test_images_2")
IMAGES_DIR = os.path.join(SCRIPT_PATH, "..", "..", "results_analysis", "test_images_2")
OUTPUT_DIR = os.path.join(SCRIPT_PATH, "..", "..", "output", "predictions_transformer_test_images_2")


def get_results(txt_file):
    bboxes = []
    probabilities = []
    with open(txt_file) as fp:
        for line in fp.readlines():
            data = list(map(float, [i for i in (line.split(","))]))
            if len(data) > 5:
                print(f"data: {data}")
                probability = data[:2]
                print(f"probability: {probability}")
                if len(probability) > 1:
                    bbox = data[2:]
                    bbox = list(map(lambda x: int(x), bbox))
                    bboxes.append(bbox)
                probability = list(map(lambda x: float(x), probability))
                probabilities.append(probability)
    return probabilities, bboxes


def get_overlay_img(img, lines_alpha=0.4):
    cv2.line(img, (0, 512), (1000, 512), (255, 255, 255), 1)
    cv2.line(img, (512, 0), (512, 1000), (255, 255, 255), 1)

    overlay = img.copy()

    for i in range(1, math.ceil(img.shape[0]/256)):
        cv2.line(overlay, (i * 256, 0), (i * 256, img.shape[1]), (255, 255, 255), 1)
        cv2.line(overlay, (0, i * 256), (img.shape[0], i * 256), (255, 255, 255), 1)
    lines_alpha = 0.4  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    overlay_img = cv2.addWeighted(overlay, lines_alpha, img, 1 - lines_alpha, 0)
    return overlay_img


def get_text_box_loc(bbox, text_box_height):
    x_min = bbox[0]
    y_min = bbox[1]-text_box_height
    x_max = bbox[2]
    y_max = bbox[1]
    if bbox[1]-text_box_height < 0:
        bbox_width = bbox[2]-bbox[0]
        if bbox[0]-bbox_width > 0:
            x_min -= bbox_width
            y_min = bbox[1]
            x_max = bbox[0]
            y_max = y_min+text_box_height
        else:
            x_min = x_max
            y_min = bbox[1]
            x_max = x_min+bbox_width
            y_max = y_min+text_box_height
    return x_min, y_min, x_max, y_max


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    total_images = 0
    for prediction_file in os.scandir(PREDICTION_RESULTS_DIR):
        total_images += 1
        file_name_with_ext = os.path.splitext(prediction_file.name)[0]
        img_name = f"{file_name_with_ext}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist, skipping ...")
            continue
        probabilities, bboxes = get_results(prediction_file.path)
        print(prediction_file.path)
        print(probabilities)
        print(bboxes)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to open image {img_path}")
            continue
        overlay_img = get_overlay_img(img)
        box_width = 92
        box_height = 20
        index = 0
        for sample_probability, sample_bbox in zip(probabilities, bboxes):
            print(f"sample_probability: {sample_probability}")
            if len(sample_probability) > 1:
                proba_str = f"{str(round(sample_probability[0], 3))}/{str(round(sample_probability[1], 3))}"
                print(proba_str)
                print(sample_bbox)
                text_box_loc = get_text_box_loc(sample_bbox, box_height)
                #print(text_box_loc[3]-text_box_loc[0])
                cv2.rectangle(overlay_img, (text_box_loc[0],text_box_loc[1]), (text_box_loc[2], text_box_loc[3]), (255, 255, 255), -1)
                cv2.putText(overlay_img, proba_str, (text_box_loc[0]+5,text_box_loc[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (120, 120, 120), 1)
                cv2.rectangle(overlay_img, (sample_bbox[0],sample_bbox[1]), (sample_bbox[2],sample_bbox[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), overlay_img)
    print(f"Total images processed for prediction: {total_images}")
