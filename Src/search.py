#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:31:37 2019

@author: aayush
"""

import ColorDescriptor
import Searcher
import argparse
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image as im


# creating the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
args = vars(ap.parse_args())

#intializing the color descriptor
cd = ColorDescriptor.ColorDescriptor((8,12,3))

#loading the query image and describe it
query = cv2.imread(args["query"])

#extract query image feature[color]
queryFeatures = cd.describe(query)

#performing the search
s1 = Searcher.Searcher(args["index"])

results = s1.search(queryFeatures)


#displaying the query
qimage = im.fromarray(query)
qimage.save('/content/Content-Based-Image-Retrieval/Src/output/query.jpg')


#loop over the results
i = 0
for (score, resultID) in results:
    #load the result image and display it
    result1 = cv2.imread(args["result_path"] + "/" + resultID)
    result = cv2.resize(result1,(300,300))
    data = im.fromarray(result)
    data.save('/content/Content-Based-Image-Retrieval/Src/output/result_%s.jpg'% i)
    i += 1
    # cv2_imshow(data)
    cv2.waitKey(0)
