# Dataset Description

These are found in the `data` directory.

images_training: directory of JPG images of 61578 galaxies. Files are named according to their GalaxyId.

training_solutions.csv: Probability distributions for the classifications for each of the training images.

images_test: directory of JPG images of 79975 galaxies. Files are name according to their GalaxyId. You will provide probabilities for each of these images. 

The first column in each solution is labeled GalaxyID; this is a randomly-generated ID that only allows you to match the probability distributions with the images. The next 37 columns are all floating point numbers between 0 and 1 inclusive. These represent the morphology (or shape) of the galaxy in 37 different categories as identified by crowdsourced volunteer classifications as part of the Galaxy Zoo 2 project. These morphologies are related to probabilities for each category; a high number (close to 1) indicates that many users identified this morphology category for the galaxy with a high level of confidence. Low numbers for a category (close to 0) indicate the feature is likely not present. 
