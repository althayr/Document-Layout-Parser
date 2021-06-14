import cv2
import torch
import datasets
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detectron2.structures import ImageList
from layoutlmft.models import layoutxlm
from layoutlmft.models.layoutxlm import (
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
)
from layoutlmft.data.utils import load_image

from config import *

# To avoid gpu issues
torch.backends.cudnn.enabled = False

tokenizer = LayoutXLMTokenizer.from_pretrained(TOKEN_CLASSIFICATION_MODEL_PATH)
model = LayoutXLMForTokenClassification.from_pretrained(TOKEN_CLASSIFICATION_MODEL_PATH)


def parse_image(image):
    width, height = image.shape[:2]
    w_scale = 1000 / width
    h_scale = 1000 / height

    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    ocr_df = ocr_df.dropna().assign(
        left_scaled=ocr_df.left * w_scale,
        width_scaled=ocr_df.width * w_scale,
        top_scaled=ocr_df.top * h_scale,
        height_scaled=ocr_df.height * h_scale,
        right_scaled=lambda x: x.left_scaled + x.width_scaled,
        bottom_scaled=lambda x: x.top_scaled + x.height_scaled,
    )

    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    return ocr_df


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def prepare_for_model(
    image,
    words,
    boxes,
    actual_boxes,
    tokenizer,
    max_seq_length,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
):
    width, height = image.shape[:2]

    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[
            : (max_seq_length - special_tokens_count)
        ]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # assert len(label_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length
    assert len(token_actual_boxes) == max_seq_length

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


def make_forward_pass(image, ocr_df, model, tokenizer):

    # Input preprocessing
    height, width = image.shape[:2]
    words = list(ocr_df.text)
    coordinates = ocr_df[["left", "top", "width", "height"]]
    bboxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        bbox = [
            x,
            y,
            x + w,
            y + h,
        ]  # we turn it into (left, top, left+widght, top+height) to get the actual box
        bboxes.append(bbox)

    normalized_bboxes = []
    for box in bboxes:
        normalized_bboxes.append(normalize_box(box, width, height))

    (
        input_ids,
        input_mask,
        segment_ids,
        token_boxes,
        token_actual_boxes,
    ) = prepare_for_model(
        image=image,
        words=words,
        boxes=normalized_bboxes,
        actual_boxes=bboxes,
        tokenizer=tokenizer,
        # max_seq_length=768)
        max_seq_length=tokenizer.max_model_input_sizes["layoutxlm-base"],
    )

    input_ids = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(input_mask, device=DEVICE).unsqueeze(0)
    token_type_ids = torch.tensor(segment_ids, device=DEVICE).unsqueeze(0)
    bbox = torch.tensor(token_boxes, device=DEVICE).unsqueeze(0)
    inference_image = ImageList.from_tensors(
        [torch.tensor(image.copy(), device=DEVICE).unsqueeze(0)], 32
    )

    # Actual forward pass
    outputs = model(
        input_ids=input_ids,
        bbox=bbox,
        image=inference_image,
        attention_mask=attention_mask,
        token_type_ids=None,
    )

    # Decoding results
    token_predictions = (
        outputs.logits.argmax(-1).squeeze().tolist()
    )  # the predictions are at the token level
    word_level_predictions = []  # let's turn them into word level predictions
    final_boxes = []
    for id, token_pred, box in zip(
        input_ids.squeeze().tolist(), token_predictions, token_actual_boxes
    ):
        if (tokenizer.decode([id]).startswith("##")) or (
            id
            in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        ):
            # skip prediction + bounding box
            continue
        else:
            word_level_predictions.append(token_pred)
            final_boxes.append(box)
    return word_level_predictions, final_boxes


def display_infered_image(image, word_level_predictions, final_boxes):
    colored = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(colored)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.load_default()

    label2text = {
        0: "O",
        1: "B-QUESTION",
        2: "B-ANSWER",
        3: "B-HEADER",
        4: "I-ANSWER",
        5: "I-QUESTION",
        6: "I-HEADER",
    }

    label2color = {
        0: "violet",
        1: "blue",
        2: "green",
        3: "orange",
        5: "blue",
        4: "green",
        6: "orange",
    }

    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = label2text[prediction]
        draw.rectangle(box, outline=label2color[prediction])
        draw.text(
            (box[0] + 10, box[1] - 10),
            text=predicted_label,
            fill=label2color[prediction],
            font=font,
        )

    return pil_img


def infer_example(image, ocr_df):
    """Runs entire pipeline for the given image (OCR + boxes inferences)"""
    word_level_predictions, final_boxes = make_forward_pass(
        image, ocr_df, model, tokenizer
    )
    pil_img = display_infered_image(image, word_level_predictions, final_boxes)
    return pil_img
