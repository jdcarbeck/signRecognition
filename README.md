# Blue Sign Location & Recognition

## Location of Blue Signs

The first task is to extract white box signs within the image. Using the image in the test originals directory, an algorithm was developed to create bounding boxes around white bordered signs which will then be attempted to be recognised. Its execution is described below.

First, the colour image is split into 3 channels. The red channel is selected and inverted. Unlike the blue channel, which has high values for white in the image, the red channel will have very low values for blue parts of the image. Inverting the red channel we get very high values for parts of the image that are blue. This will give a good contrast between the white of the sign and the blue as the white aspects of the sign will have relatively higher values in the red channel. The brightness of the colour image was also reduced by 50 to mitigate strong lighting on blue signs.



![](https://lh3.googleusercontent.com/60E6qI9QmOuQaJ2sdYg9QN_Aa130cgP4WK6W6TyFMJX_F7Unopbk_wucs05puuyglhAG8fwCZRThOmcJQAsYZt-InHLP416q3ZIq_xoBs1Hde0Y0bh81zflCg0twgIdwJWULlqUC)

![](https://lh3.googleusercontent.com/kEBIMQyBcG8w-qiF6XROh_-RaCWP8fmLDGhGXh6VJ11MyOH3i3Q4HB7VwlsW1-EORW4iw6QEOs8_DgYkB3wXT9I-BKWDC50d-poPHJyCZnqNE8ehZL94qgdTLIGv2kCFJUXSTMvl)

Comparison between an RGB image and Inverted Red channel image

![](https://lh3.googleusercontent.com/2QlOzkbTmd5jXKacqYx8fBuRIYUtUIzAF-D6LqegF2GbF-mBFIxwIbtFkJxHxekc-NVs_OhJJbvU9rMhe_IaCGNlNX0fpTeFT4AdmksaZaNVePjv-9ZbbS8SJHMnhbCapNyXxXUj)
Adaptive Thresholding after a 5x5 3 iteration opening

Using adaptive binary thresholding and morphology we are able to get a clear binary image of the blue signs.

Adaptive binary thresholding handles the different lighting conditions across the image by dynamically setting the threshold based on the segment of the image that is being thresholded. Using the grayscale image from the red channel the adaptive threshold was done using a blockSize of 75x75, a C value of 2, and gaussian weighted sums of the neighbourhood values. Gaussian weighted adaptive thresholding provided a cleaner image. Which proved better results then mean weighted adaptive thresholding.

The binary image was then cleaned via mathematical morphology. An opening morphology was used 3 times on the image using a kernel size of 5. This was done to close sign borders that were 0 in the binary thresholded image. From the morphology we get contiguous edges for the white border around the sign. Morphing the image also reduced the noise of images which providing fewer and cleaner edges that were detected in further steps.


![](https://lh6.googleusercontent.com/a8T0PqyHWu8alViOANKjXED8oeDAApZR6_IiFzoWdIa9CpsfkUimxlUuBiu00RLkTp_NAQWgm04dMZh5EnhlJO6c-g7XLNPnhFIekGUZWQHj4STvc89uBDminslklj7UMfbzjMo-)

![](https://lh6.googleusercontent.com/xFZis_R8UkafL49vUvsCBCxHS2PkqLpJ0gN2-PEUG-HzNzU2IA0SuFUKoD-TjAkRVsBTpgcW-PPEQ8GSOklFymL_YXDGiDyAPbjfB_JxIAODVgTH-Dbi_w7GBUwA45ghrbVKpGlS)
Examples of the thresholded and morphed image

Edges are then drawn using canny edge detection. Canny doesn’t provide anymore information than 1st derivative edge detection using Sobel, but is easily implemented via OpenCV.

From the edge image, contours are found. For each found contour a bounding box is created. The bounding box contains all of the contour points in a minimum area rectangle. This bounding box is used to represent a detected point of interest. The points of interest of the image are filtered via a comparison of their bounding boxes to the contour used to create the bounding box. Using the contour area and the bounding box area contours are selected as being possible signs if the contour area covers 60% of the bounding box, area of the contour is greater than 5000 pixels, and if the contour has a child and doesn't have a parent.

Area is used as the square signs should have a bounding box relatively close in area to that of the square sign. Setting a minimum area for the contour also reduces the amount of small contours from the noisy edge image. Using the hierarchy from finding the contour ensures the contour selected is that of the white border.

![](https://lh6.googleusercontent.com/QZhVsa8eZmW7w0-wPWYuNhE4lv5JYAAvGCbBxYTdXiD9jYBVRzTWSxOOzN3yhiOcvj7ckfEZyHxrymKPOI7xaXsd5H_BGSGfvdHbQtZ-vbJloyZb8dLRHGoD1hDOncHPcNwGesVE)
![](https://lh3.googleusercontent.com/6vfPvYNG_zXBMdn9z4s44haw8keLtStT7qdmpahGQ46ICLIBrO32V1o2vCb_-gr9GBXEH7Wp_om2D8ATXp3bJXp0c6vTpLbGCGb170s8Bl9TaCzZnQHwSsHSr6g4uda2QYW9UFSW)
Bounding Boxes on Points of Interest

For each of the points of interest the sign dimensions will be extracted and attempted to be recognised in the addSignObject. As seen in the images above and below a small amount of non signs segments will be detected as points of interest, these will be discarded in addSignObject method.


### Discussion

The implementation above does suffer from poor lighting conditions. In edge cases with poor lighting the white values in the red channel can be low enough to not have enough contrast with the blue values in the red channel. This scenario causes the image to not be properly thresholded, missing the white borders of the sign. The same is true in cases where there is a lot of light on the blue signs. In this case the blue sign pixels value in the red channel would be higher and can in cases not have enough contrast with the whilte values in the red channel to be properly thresholded.

Morphing the image does reduce the view distance of recognition of signs but is necessary from the thresholding inorder to open the border of the binary image so it is contiguous. 

Filtering of contours could be more selective. The above implementation does produce higher false positives for signs in an attempt to reduce the amount of false negatives for signs. Taking the ratio of width and height of the bounding boxes would allow for further reduction of the points of interest. Using the hierarchical values does cause some issues where the whole sign is detected rather than the boxes within the blue sign. To prevent this further processing on the selected point of interests could be done. 


## Recognition


### Approach

Each point of interest is passed to a method addSignObject which takes the original image and a given point of interest and attempts to find the corners that define the sign object. Using the bounding rectangle the image is copied and cropped for further processing. The next steps are determining the corners of the sign for geometric transformation and then recognition via template matching. 

The original image is cropped to the bounding box and a border is added around the image to correct any overcropping of the image that lost the white border. The image is then processed similarly to when locating the blue signs. The red channel is thresholded, morphed, then edge detected. Instead of adaptive thresholding Otsu thresholding is used. This reprocessing of the image was done to better extract the features of the blue sign, as disregarding background from the original image was a previously a concern.

Using the edge image contours are found. Using the contours a polygon is drawn around contours that have children and no parent, examined via the hierarchy of contours. A polygon is used to find the bounds of the white box. An epsilon value is generated by taking 8% of the contour length. The epsilon value is used to ignore unclean edges of signs from the binary image. This makes the approximate polygon most commonly four sided.

Assuming that the contour being selected is that of the white box, the first four corners of the approximate polygon are then stored in an array of points. The polygon points are then analyzed to detect which of the corners is the top left, top right, bottom right, and bottom left. This is done by determining the sum and difference of the x and y values of each point.

![](https://docs.google.com/drawings/u/1/d/sN6gpBs0rx_c9feT6eECRJg/image?w=509&h=271&rev=131&ac=1&parent=1A5QtX7pXimlj-8lkVPMiFjVQddK8EnWDlHBPXT4ztcQ)

The top left corner with be the minimum sum of the value pair where the bot right will be the max sum of the value pair. The top right will be the maximum difference of the value pair where the bot left will be the minimum difference of the value pairs. 

![](https://lh4.googleusercontent.com/cy1CMzLHAMEOoVB7y_1LlEIZZtlPD3Wi61Pv_NIwt9OLZxk2aOmwlxcxCKaHXD6GJQUt7q7fBK4x8r3rI45IlM-p5Y44NBUPDSvLxqktZ1oDOka1oGN6Eam8QPblV7MOzb3qwlfB)

![](https://lh5.googleusercontent.com/YBfXIHivrUGCxOClUOy0MDvW7E31fZ4XeP40DlrW8idxobuRjv2k0-L3KG4_7tKFZegepQMlNUjNyck736_qKCAz2B3zKcplJVGahiTGXXp9QA-C1tpSIPZf90wfwIIET11OeLNg)
![](https://lh4.googleusercontent.com/TcEXzQY40rtyUXtk1S5itemX3J5J5Fv1kMCL4lAAW8riT8uC-f2wSb0ZREWvcNQGaNPcOFZDocq08CgWBw6K8EiEAuFCHRHXWTF0xo4z6SVjf7JlSXtiCPv8DLT5INapbsHWTVxN)
Corner detection of signs: (Red = TL, Green=TR, Blue=BR, Yellow=BL)

With the corners found the image can be geometrically transformed so the sign becomes square. The found corners are added to the top left coordinates of the crop to get there location in regards to the original image.

The destination points are a 10% offset from the bounds of the warped result matrix. This is to ensure that the whole sign is included for template matching. The image is then warped to the resulting matrix.

![](https://lh3.googleusercontent.com/q4zasdAInUHoJgE7oKGwHm116vgxVbX3SwDMOJd7Mz6GR9mRvWLMOWCc6daArTualtt2jkfFqE46b425qWE9h_IbpdhP2CSAtLZaSseR9-yO1OFDxn3XmLTy1JvdLUXq8-chx5_b)
Collection of geometrically transformed signs based of corner recognition

Every found object in an image it is then compared with each of the training images annotated objects. A minimum value and the name of the object that gave that value is maintained across all training images. Each object image is passed to each of the Annotated images method findBestMatch which compares each object in the annotated image to that of the passed image.

The images are compared in the findBestMatch method using the compare objects method. Here the passed object is the img and the annotated image object is the template. From the geometric transformation the sign should take up about 90% of the image. The annotated object is 100% so we scale down this template image and vary its scale when we perform template matching using a normalised square difference as the output. This will give us a value between 1 and 0 in the matching space based off the template. Where the minimum value is the at the location with the best match.

The minimum value found  is compared and changed based off all the annotated images. The lowest value’s name after all iterations is then set to the unidentified objects name.

![](https://lh4.googleusercontent.com/PPFkRNlXm-hCtroZw3JNAtRdcAI0GRLalLVjNjQNglnpn-V9PspjDaL6ZymusBYwMnDL8BFFjyNgAvYO1uh6nGCct2UJoPYdNJ6TH4lJv0_wmdIsxLpUI3fAsYzsJC2T3x0Nu3mT)
Recognition Example A

![](https://lh5.googleusercontent.com/lf6RuiiW8CK_r2ylwNgkOqfl8_QO_mA4QYifgvfgYsTavhBkUR479Um4bzEX8mt9dECX8ECqsN-YloPG8Ivzrz-cJ-DgNIAd3Mfl7I1M8F0eo2zAzPapaAm0s5fekhFjJiJrNU18)
Recognition Example B

![](https://lh5.googleusercontent.com/o3a_2OTB3SyL84f4GA_LcKj2N2jajxNi0VDfv0DKf3paNvDsZwCia4f4x4ABFwega7Q8gpiZK7fjJxKXCDyAYP2s90wKg8dguA4ymLsfUZ3tbYDl2QiE9aSbBIy8GmgGTMt1UWrH)
Recognition Example C


### Discussion

The main point of improvement can be made in the geometric transformation of points of interest into objects. Very few signs are miss selected as points of interest but significant portion of them are discarded. This has to do with corner detection. It might prove better to look for extreme changes on the contour to determine corners. This might reduce the amount of images that are discarded due to poor corner recognition. Approximating the polygon breaks down when the edges in the edge image are not well representative of the white lines of the border. Tweaking of the corner recognition would definitely improve the recall of the recognition. 

Template matching is pretty successful but other methods such as feature extraction could improve recognition. Feature extraction recognition was attempted and code exists for adding features to a vector but a successful svm model or statistical pattern recognition model were unable to be generated. A larger set of ground truth may allow for implementations of these to be successful.

Template matching does fail in some cases where the templates are similar, such as the case for the one and gents sign. Nearly half of the ones were recognized as gents. In cases such as this where the templates are quite similar, the minimum values found could be discounted via feature vector comparison.

The recognition here also could do better with determining unknown signs. Some of the images in the testing set include white box signs that are not part of the training set. In this implementation of template matching determining unknown values is difficult due to the similarity between templates. A feature based approach for classification might have better results as the feature space should be selected to be able to distinguish even between similar signs such as ladies and gents.



## **Results**

![](https://lh5.googleusercontent.com/YEkAZ22KkDJuwrwEu7U3tQ6Xh-_-JZvLl7mlO-aJNhkkKyMvkecm0DnPnZqBrWb3LZBsY9CpGe3pBdWxbyiKB6FRz6hp5OelmrTqyaIC5yVlnxe7UOkyeZcAzF-Iu_glAEZwhcpa)

Precision: 0.716216

Recall: 0.654321

F1 = 0.683871


### Discussion

This implementation is moderately successful, but two aspects of the implementation are mostly responsible for lower than ideal precision and recall. The recall of the implementation comes down to issues with the corner extraction. Using more robust ways of extracting corners such as finding extremes in a contour for corner detection might reduce the amount of points of interest that do contain signs from failing to be extracted. A feature based classification could prove to have a better precision rate. This would include a better implementation for disregarding signs not in the training set. An addition to the existing code that looks at a feature vector for similarly matched templates could greatly improve precision. As in the table where ones were recognised as gents as well as exits being recognised as gents.


### Failed Attempts

Due to the poor performance of template matching, other attempts at classification were made. There is code implemented to set and read the feature vector for a found object. A first attempt was a simple classification method that used 2 feature sets, child contours and first child contour area, in a support vector machine. This feature space was not descriptive enough to get multi-class classification. Another attempt was made using a support vector machine using a histogram of orientated gradients, but was also not able to surpass template matching performance. Statistical pattern recognition was not attempted due to the small data set. This might not have been true with a well selected feature vector.
