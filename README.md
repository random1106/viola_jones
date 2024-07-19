# Viola Jones face-detection algorithm

We implement the algorithm of the paper [Viola & Jones 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) for face-detection. 
The data we used is described at [here](http://cbcl.mit.edu/software-datasets/FaceData2.html), and can be downloaded from 
[here](www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz). Each image is of 19 * 19 grayscale pixel. The training dataset contains 2429 faces and 4548 non-faces. 
The testing dataset contains 472 faces and 23573 non-faces.

We obtain 95.7% correctness on the training dataset and 95.2% correctness on the testing dataset.
