import logging
from typing import List


def get_iou(bbox1: List[int], bbox2: List[int]) -> (float, float, float):
    """
    Returns IoU percentage

    :param bbox1: first bounding box
    :type bbox1: a list containing the top-left coordinate of the bounding box and its width and height
    :param bbox2: second bounding box
    :type bbox2: a list containing the top-left coordinate of the bounding box and its width and height
    :return: a tuple consisting of percentage of overlap between the two bounding boxes, and bounding boxes areas.
    """
    max_left_x = max(bbox1[0], bbox2[0])
    max_top_y = max(bbox1[1], bbox2[1])
    min_right_x = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    min_bottom_y = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    # In case that two bounding boxes do not interest exit the function
    if min_right_x < max_left_x or min_bottom_y < max_top_y:
        return 0.0, None, None

    # Two bounding boxes interesection area
    interest_area = (min_right_x - max_left_x) * (min_bottom_y - max_top_y)

    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    iou = interest_area / (bbox1_area + bbox2_area - interest_area)

    return iou, bbox1_area, bbox2_area


def filter_overlapping_predictions(predictions: List[List[float]], threshold_iou=0.5, probability_precision: int=3) -> List[float]:
    """
    Filters out all bounding boxes that overlap with other bounding boxes for more than 50% and
    that have the lowest probability that they contain school compunds.

    :param predictions: a list of predictions and bounding box coordinates with the following format:
        1. binary_classifier_probability
        2. localizer_probability
        3. bounding_box_top_left_x_coordinate
        4. bounding_box_top_left_y_coordinate
        5. bounding_box_bottom_rigth_x_coordinate
        6. bounding_box_bottom_right_y_coordinate
    The list item can have a single value in case when there are no bounding boxes predictions.
    :param threshold_iou: two bounding boxes overlap threshold
    :param probability_precision: precision in decimal digits when rounding the probability
    :return: a list containing non-overlapping bounding boxes and overlapping bounding boxes that have the
    highest probability that they contain school compunds.
    """
    discarded_bboxes = []
    predictions_len = len(predictions)
    logging.debug(f"Total predictions: {predictions_len}")
    logging.debug(f"Predictions: {predictions}")
    for i in range(0, predictions_len-1):
        p1 = predictions[i]
        if len(p1) == 1:
            continue
        if p1 not in discarded_bboxes:
            # if there is no localizer prediction skip bbox
            if len(p1) == 5:
                continue
            for j in range(i+1, predictions_len):
                p2 = predictions[j]
                if len(p2) == 5:
                    continue
                if p2 not in discarded_bboxes:
                    bbox1 = p1[2:]
                    bbox2 = p2[2:]
                    logging.debug(f"bbox1: {bbox1}, bbox2: {bbox2}")
                    iou, bbox1_area, bbox2_area = get_iou(bbox1, bbox2)
                    logging.debug(f"iou: {iou}")
                    if iou >= threshold_iou:
                        bin_prob1 = round(p1[0], probability_precision)
                        bin_prob2 = round(p2[0], probability_precision)
                        if bin_prob1 == bin_prob2:
                            loc_prob1 = round(p1[1], probability_precision)
                            loc_prob2 = round(p2[1], probability_precision)
                            prediction_to_discard = p1 if loc_prob1 < loc_prob2 else\
                                                    p2 if loc_prob1 > loc_prob2 else\
                                                    p2 if bbox1_area < bbox2_area else\
                                                    p1
                            logging.debug(f"Discarding prediction: {prediction_to_discard}")
                            discarded_bboxes.append(prediction_to_discard)
                        else:
                            prediction_to_discard = p1 if bin_prob1 < bin_prob2 else p2
                            logging.debug(f"Discarding prediction: {prediction_to_discard}")
                            discarded_bboxes.append(prediction_to_discard)
    logging.debug(f"Total discarded bounding boxes: {len(discarded_bboxes)}")
    kept_bboxes = [p for p in predictions
                   if p not in discarded_bboxes]
    return kept_bboxes


