#!/usr/bin/env python3
import argparse
import os
import sys

import keras
import numpy as np
import cv2

import PIL
from snark.imaging import cv_image
from skimage.io import imsave

def resize(image, output_shape):
    """Fast resize using PIL (ensure pillow-simd is installed instead of pillow)"""
    image = PIL.Image.fromarray(image)
    image = image.resize(output_shape[:2])
    return np.array(image)

def load_model(src_dir):
    from keras.models import model_from_json
    with open(os.path.join(src_dir, "model.json"), "r") as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(os.path.join(src_dir, "weights.h5"))
    return model


def cv_cat_gen(args, stream=sys.stdin.buffer):
    for pair in cv_image.iterator(stream):
        image = pair.data
        if args.crop:
            y1, x1, y2, x2 = args.crop
            image = image[y1:y2, x1:x2, ...]
        if args.bayer == 'BG':
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        elif args.bayer:
            raise ValueError("Unknown bayer value")
        if args.uint8 is not None:
            image = (image.astype(np.float32) * args.uint8).astype(np.uint8)
        if image.shape[:2] != args.input_shape:
            image = resize(image, args.input_shape[:2])
        yield pair.header, image

def do_predict(model, batch, batch_headers, args):
    images = np.array(batch, dtype=np.uint8)
    predicted = model.predict(images)[..., 1]
    for head, pred, image in zip(batch_headers, predicted, images):
        timestamp_str = head[0].item().strftime('%Y%m%dT%H%M%S.%f')
        print("{:s},{:.2f}".format(timestamp_str, pred), file=sys.stdout)
        if pred > args.output_confidence > 0:
            imsave(timestamp_str + ".jpg", image)

def main(args):
    print("Loading model:", args.model_dir, file=sys.stderr)
    model = load_model(src_dir=args.model_dir)
    batch = []
    batch_headers = []
    for header, image in cv_cat_gen(args):
        # print("<<", header, file=sys.stderr)
        batch.append(image)
        batch_headers.append(header)
        if len(batch) == args.batch_size:
            do_predict(model, batch, batch_headers, args)
            batch = []
            batch_headers = []
            sys.stdout.flush()
    if batch: # leftovers
        do_predict(model, batch, batch_headers, args)

def get_args():
    '''Get args from the command line args'''
    parser = argparse.ArgumentParser(
        description="Takes image paths from STDIN and outputs to STDOUT the confidence that they contain a Macbeth checkerboard.")
    parser.add_argument("model_dir", type=str, help="The directory that houses model.json and weights.h5")
    parser.add_argument("--batch-size", type=int, help="Batch size to use on GPU, higher gives slightly more performance", default=10)
    parser.add_argument("--input-shape", type=str, help="Model input shape in CSV format (default '224,224,3').", default="224,224,3")
    parser.add_argument("--bayer", type=str, help="Debayer raw image {BG}")
    parser.add_argument("--uint8", type=float, default=None, help="Convert 16 bit to 8 bit by multiplying by this constant (usually 1/255)")
    parser.add_argument("--crop", type=str, default=None, help="Crop image to y1,x1,y2,x2")
    parser.add_argument(
        "--output-confidence",
        type=float, default=-1,
        help="Output images with confidence greater than this. -1 means don't output (default).")
    args = parser.parse_args()
    args.input_shape = tuple([int(i) for i in args.input_shape.split(",")])
    if args.crop is not None:
        args.crop = tuple([int(i) for i in args.crop.split(",")]) 
    return args

if __name__ == "__main__":
    main(get_args())
