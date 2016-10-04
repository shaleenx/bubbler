#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse


# argument parser for parsing image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# integer mappings of the question numbers
# to the index of the correct bubble
# (zero indexed)
ANSWER_KEY = {0:1, 1:2, 2:3, 3:4, 4:1}

# image pre-processing
image = cv2.imread(args['image'])   # load the image
im_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # convert it to grayscale
im_blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # blur it slightly to reduce high frequency noise
im_edged = cv2.Canny(blurred, 75, 200)  # find edges

# find contours in the edge map and initialize the contour
# that corresponds to the document
contours = cv2.findContours(im_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
docCnt = None

# ensure that atleast one contour was found
if len(contours) > 0:
    # sort the contours according to their size in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to both the original image and the grayscale image
# to obtain a top-down birds ee view of the paper
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(im_grayscale, docCnt.reshape(4,2))


