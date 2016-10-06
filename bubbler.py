#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
from imutils import contours
from imutils.perspective import four_point_transform
import imutils

def order_points(pts):
    '''initialize a list of coordinates that will be ordered such that they are in the order
    of top left, top right, bottom-right, bottom-left.'''
    rect = np.zeros((4,2), dtype='float32')

    # the top-left point will have the smallest sum, whereas the bottom right point will
    # have the largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the top-right point will hav the
    # least difference, whereas the bottom-left will have the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    return rect # the ordered coordinates

def four_point_transform_my(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = order_points(pts)

    # compute the width of the new image, which will be the maximum distance between
    # bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the maximum distance between the
    # top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct the set of destination points
    # to obrain a birds eye view (top-down view) of the image, again specifying points in the
    # top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([ [0,0], [maxWidth-1,0], [maxWidth-1,maxHeight-1], [0, maxHeight-1]], dtype='float32')

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

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
im_blurred = cv2.GaussianBlur(im_grayscale, (5, 5), 0)  # blur it slightly to reduce high frequency noise
im_edged = cv2.Canny(im_blurred, 75, 200)  # find edges

# find contours in the edge map and initialize the contour
# that corresponds to the document
cnts = cv2.findContours(im_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# ensure that atleast one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts:
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

# apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize the list of contours that
# correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the bounding box
    # to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w/float(h)

    # in order to label the contour as a question, region should be sufficiently wide,
    # and sufficiently tall, and have an aspect ratio of almost 1
    if w>=20 and h>=20 and ar>=0.9 and ar<=1.1:
        questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize the total number of correct answers
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# each question has 5 possible answers, to loop over the question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from left to right, then
    # initialize the index of the bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None

    # loop over the sorted contours
    for (j,c) in enumerate(cnts):
        # construct a mask that reveals only the current "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the threshold image, then count the number of non-zero
        # pixels in the bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total non-zero pixels, then
        # we are examining the currently bubbled-in answer
        if bubbled is None or total>bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and the index of the "correct" answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    # check to see if the bubbled answer is correct actually
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

print("No. of Correct Answers:", correct)
# scoring the exam and displaying the results to the screen
score = (correct/5.0)*100
print("Score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)

