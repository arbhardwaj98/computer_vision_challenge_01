Congratulations, you have been selected to advance to the coding challenge phase.

If you are receiving this challenge, then you have left a very good impression in your first conversation with the Hyperspec AI team. We want you to now prove some of your coding chops using your computer vision skills.

The goal of this challenge is to pick the needle in a haystack.

In the query_images folder, there are 1,331 image consisting of various spatial transformations (translation and rotations). Each image is represented by a SHA1 hash, so the images are randomly sorted. 

In this challenge you have to find the best match between the query images and the single aerial image. To further clarify, you have to pick one query image which matches the most with the aerial image on a pixel by pixel basis. This means that you have to iterate through each image and find the image that most closely aligns with the aerial image on a pixel by pixel basis. We show you the right answer in merged.png. Once you find the best match, please send us the image filename that you think is the best match and the corresponding algorithm that you used to find the best image candidate.


### SOLUTION:

The best matching image is `c1e2150520cb798ec63a7e8d0f311e54aba92484.png`. The instructions
for setting up and running the code are given in the following sections.

#### Setup:
Create and initialize a conda environment for the project: 
```
$ conda create -n cv_challenge_1 python=3.7
$ conda activate cv_challenge_1
```
Install the required python packages:
```
$ pip install -r requirements.txt
```

#### Testing:
To start the query simpy run the following command:
```
$ python3 main.py --RAY_verbose_spill_logs=0
```
The code saves an overlay of the best matching image on the reference image as `best_match_overlay.png`


