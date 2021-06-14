import cv2
import argparse
import pytesseract
import numpy as np
from config import *

def parse_image(img):
    
    if isinstance(img, str):
        image = cv2.imread(img)
    else:
        image = img.copy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, DEFAULT_IMG_SIZE)

    width, height = image.shape[:2]
    w_scale = 1000/width
    h_scale = 1000/height
    

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame') \
            
    ocr_df = (ocr_df
              .dropna()
              .assign(left_scaled = ocr_df.left*w_scale,
                      width_scaled = ocr_df.width*w_scale,
                      top_scaled = ocr_df.top*h_scale,
                      height_scaled = ocr_df.height*h_scale,
                      right_scaled = lambda x: x.left_scaled + x.width_scaled,
                      bottom_scaled = lambda x: x.top_scaled + x.height_scaled)
              .sort_values(by=['top', 'left']))

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    return ocr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--dst", type=str, default="ocr.csv")
    args = parser.parse_args()
    df = parse_image(args.image)
    df.to_csv(args.dst)
