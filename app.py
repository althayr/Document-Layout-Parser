import cv2
import logging
import gradio as gr
import numpy as np
from pdf2image import convert_from_path

# Project imports
from config import *
from image_processing import find_and_normalize_document, bw_scanner
from ocr_layer import parse_image
from predict import infer_example


def read_image(path):
    logging.debug(f"Image path : {path}")
    image = None
    try:
        logging.debug("Trying to read image file")
        image = cv2.imread(path)
        if image is not None:
            return image
        # if cv2 fails to read image
        else:
            raise ValueError
    except Exception:
        logging.debug("Failed to read as image")
        pass

    # Try reading as PDF
    if not image:
        try:
            logging.debug("Trying to read image file as PDF")
            pages = convert_from_path(path, 500)
            if not len(pages) > 1:
                image = np.asarray(pages[0])
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception:
            logging.debug("Failed to read as PDF")
            pass

    if not image:
        raise ValueError("Failed to read file as image or pdf!")


def display_receipt_ocr(image_path, background_removal):

    img = read_image(image_path)
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if background_removal:
        logging.debug("Removing image background")
        img = find_and_normalize_receipt(img=img)
    else:
        img = bw_scanner(img)
    df = parse_image(img)
    image = cv2.resize(img.copy(), DEFAULT_IMG_SIZE)
    infered = infer_example(image, df)

    # Exporting resulting files
    df.to_csv(OUTPUT_OCR_CSV)
    cv2.imwrite(OUTPUT_NORMALIZED_IMG, img)
    cv2.imwrite(OUTPUT_INFERED_IMG, np.asarray(infered))

    return (
        original,
        img,
        infered,
        OUTPUT_OCR_CSV,
        OUTPUT_NORMALIZED_IMG,
        OUTPUT_INFERED_IMG,
    )


if __name__ == "__main__":

    sample_images = [
        ["./images/fax_form.png", False],
        ["./images/groceries_1.jpg", True],
        ["./images/groceries_2.jpeg", True],
        ["./images/starbucks_spanish.pdf", False],
    ]

    inputs = [
        "text",
        gr.inputs.Checkbox(
            label="Image needs background removal? (Eg: from phone camera)"
        ),
    ]
    outputs = ["image", "image", "image", "file", "file", "file"]
    iface = gr.Interface(
        fn=display_receipt_ocr, inputs=inputs, outputs=outputs, examples=sample_images
    )
    iface.launch()
