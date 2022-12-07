import os
import numpy as np
import pandas as pd
from traffic_sign_detector import TrafficSignDetector
from traffic_light_detector import TrafficLightDetector


IMAGE_FOLDER = "./image"
INPUT_FOLDER = "./image/input_image"
OUTPUT_FOLDER = "./image/output_image"

DEBUG = False

# Hyperparameters (traffic sign related)
SIGN_K_SIZE = 5
SIGN_YUV_ERROR = np.array([90, 40, 70])
SIGN_MATCHING_LOWER_BOUND = 0.0001
SIGN_MATCHING_UPPER_BOUND = 0.02
SIGN_DILATION = 15

# Hyperparameters (traffic light related)
LIGHT_K_SIZE = 5
LIGHT_CUTOUT = 0.5
LIGHT_MIN_RADIUS = 0.002
LIGHT_MAX_RADIUS = 0.01
LIGHT_YUV_ERROR = np.array([80, 60, 50])
LIGHT_DILATION = 10


def detect_traffic_sign():
    """
    Detect traffic sign for all input images
    Compute the number of correct annotation and of correct coordinates
    """

    subfolder = "traffic_sign"

    detector = TrafficSignDetector(
        k_size=SIGN_K_SIZE,
        yuv_error=SIGN_YUV_ERROR,
        matching_lower_bound=SIGN_MATCHING_LOWER_BOUND,
        matching_upper_bound=SIGN_MATCHING_UPPER_BOUND,
        dilation=SIGN_DILATION,
        debug=DEBUG
    )

    # Read the annotation csv. Prepare columns for update
    df = pd.read_csv("./annotation/traffic_sign_annotations.csv", sep=";")
    df["Predicted annotation"] = ""
    df["Predicted X"] = np.inf
    df["Predicted Y"] = np.inf

    # Read and detect each image in the input folder
    filenames = df["Filename"]
    number_of_images = len(filenames)
    for i in range(number_of_images):
        img_file = os.path.join(INPUT_FOLDER, subfolder, filenames[i])
        pred_sign, pred_coordinates = detector.run(img_file, os.path.join(OUTPUT_FOLDER, subfolder))
        if pred_sign:
            df.loc[i, "Predicted annotation"] = pred_sign
            df.loc[i, "Predicted X"] = pred_coordinates[0]
            df.loc[i, "Predicted Y"] = pred_coordinates[1]

    # Compare the prediction and actual values
    df["Correct sign"] = df["Predicted annotation"] == df["Annotation tag"]
    error = np.sqrt((df["Predicted X"] - df["Upper left corner X"]) ** 2 + (df["Predicted Y"] - df["Upper left corner Y"]) ** 2)
    df["Correct coordinates"] = error < 15 # allow 15 diagonal pixels (equivalent to 10 horizontal or vertical pixels) for error
    df["Both correct"] = np.logical_and(df["Correct sign"], df["Correct coordinates"])
    columns_to_write = ["Filename", "Annotation tag", "Predicted annotation", "Predicted X", "Predicted Y", "Upper left corner X", 
                            "Upper left corner Y", "Correct sign"," Correct coordinates", "Both correct"]
    df.to_csv("./traffic_sign_predictions.csv", sep=";")
    # print(sum(df["Correct sign"]), sum(df["Correct coordinates"]), sum(df["Both correct"]))


def detect_traffic_light():
    """
    Detect traffic light for all input images
    Compute the number of correct annotation and of correct number of lights
    """
    
    subfolder = "traffic_light"

    detector = TrafficLightDetector(
        k_size=LIGHT_K_SIZE,
        cutout=LIGHT_CUTOUT,
        min_radius=LIGHT_MIN_RADIUS,
        max_radius=LIGHT_MAX_RADIUS,
        yuv_error=LIGHT_YUV_ERROR,
        dilation=LIGHT_DILATION,
        debug=DEBUG
    )

    # Read the annotation csv. Prepare columns for update
    df = pd.read_csv("./annotation/traffic_light_annotations.csv", sep=";")
    expected_number_df = df.groupby(["Filename", "Annotation tag"]).size().reset_index().rename(columns={0: "Expected number"})
    df = pd.merge(df, expected_number_df, how='left',on=["Filename", "Annotation tag"])
    df = df.drop_duplicates(subset=["Filename"]).reset_index(drop=True)

    # Read and detect each image in the input folder
    filenames = df["Filename"]
    number_of_images = len(filenames)
    for i in range(number_of_images):
        img_file = os.path.join(INPUT_FOLDER, subfolder, filenames[i])
        pred_anno, pred_num = detector.run(img_file, os.path.join(OUTPUT_FOLDER, subfolder))
        if pred_anno:
            df.loc[i, "Predicted annotation"] = pred_anno
            df.loc[i, "Predicted number"] = pred_num

    # Compare the prediction and actual values
    df["Correct anno"] = df["Predicted annotation"] == df["Annotation tag"]
    df["Correct number"] = df["Predicted number"] == df["Expected number"]
    columns_to_write = ["Filename", "Annotation tag", "Expected number", "Predicted annotation", "Predicted number", "Correct anno", "Correct number"]
    df = df[columns_to_write]
    df.to_csv("./traffic_light_predictions.csv", sep=";")
    # print(sum(df["Correct anno"]), sum(df["Correct number"]))


if __name__ == "__main__":
    detect_traffic_sign()
    detect_traffic_light()
