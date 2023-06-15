

predicted_labels_dir = ''
ground_truth_labels_dir = ''

# Define a function that calculates a list of IOUs between a predicted box and all ground truth boxes
# Where a box is specified by a 4 tuple (TL_x, TL_y, BR_x, BR_y)
def calculate_IOUs(predicted_box, ground_truth_box):
    # calculate the area of the intersection rectangle
    intersection_area = (min(predicted_box[2], ground_truth_box[2]) - max(predicted_box[0], ground_truth_box[0])) * (min(predicted_box[3], ground_truth_box[3]) - max(predicted_box[1], ground_truth_box[1]))

    # calculate the area of both rectangles
    predicted_box_area = (predicted_box[2] - predicted_box[0]) * (predicted_box[3] - predicted_box[1])
    ground_truth_box_area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])

    # calculate the IOU
    iou = intersection_area / float(predicted_box_area + ground_truth_box_area - intersection_area)

    return iou




#############################################################################################
# THE SCRIPT
#############################################################################################
