# Description

Understanding how and why we are here is one of the fundamental questions for the human race. Part of the answer to this question lies in the origins of galaxies, such as our own Milky Way. Yet questions remain about how the Milky Way (or any of the other ~100 billion galaxies in our Universe) was formed and has evolved. Galaxies come in all shapes, sizes and colors: from beautiful spirals to huge ellipticals. Understanding the distribution, location and types of galaxies as a function of shape, size, and color are critical pieces for solving this puzzle.

![Galaxy](Galaxy)

The Whirlpool Galaxy (M51). Credit: NASA and European Space Agency

With each passing day telescopes around and above the Earth capture more and more images of distant galaxies. As better and bigger telescopes continue to collect these images, the datasets begin to explode in size. In order to better understand how the different shapes (or morphologies) of galaxies relate to the physics that create them, such images need to be sorted and classified. Kaggle has teamed up with [Galaxy Zoo](Galaxy Zoo) and [Winton Capital](Winton Capital) to produce the **Galaxy Challenge**, where participants will help classify galaxies into categories.

Galaxies in this set have already been classified once through the help of hundreds of thousands of volunteers, who collectively classified the shapes of these images by eye in a successful citizen science crowdsourcing project. However, this approach becomes less feasible as data sets grow to contain of hundreds of millions (or even billions) of galaxies. That's where you come in.

This competition asks you to analyze the JPG images of galaxies to find automated metrics that reproduce the probability distributions derived from human classifications. For each galaxy, determine the probability that it belongs in a particular class. Can you write an algorithm that behaves as well as the crowd does?

Contributors: D. Harvey, C. Lintott, T. Kitching, P. Marshall, K. Willett, Galaxy Zoo 

## Acknowledgments

The Contributors and the rest of the Galaxy Zoo and Kaggle teams would like to say a big thank you to Winton Capital for helping make this happen. Without their support, we would have not been able to make this competition go ahead.

# Evaluation

This competition uses [Root Mean Squared Error](Root Mean Squared Error) as the evaluation metric.

RMSE = sqrt( 1/N * sum((pi - ai)^2) )

Where:

\\(N\\) is the number of galaxies times the total number of responses

\\(p\_i\\) is your predicted value

\\(a\_i\\) is the actual value

## Submission Format

Your submission file must have a header and should be structured in the following format. The [Data](Data) page provides some example solution files.

GalaxyId,Class1.1,Class1.2,Class1.3,Class2.1,..., Class11.5, Class11.6
100002,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
100003,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
100004,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
etc...

## The Galaxy Zoo Decision Tree

Galaxy Zoo guides its citizen scientists through a nested decision tree - this is what constitutes the classification process. The decision tree consists of 11 questions, with each question having 2-7 responses.

List of Questions
Q1. Is the object a smooth galaxy, a galaxy with features/disk or a star? 3 responses

Q2. Is it edge-on? 2 responses

Q3. Is there a bar? 2 responses

Q4. Is there a spiral pattern? 2 responses

Q5. How prominent is the central bulge? 4 responses

Q6. Is there anything "odd" about the galaxy? 2 responses

Q7. How round is the smooth galaxy? 3 responses

Q8. What is the odd feature? 7 responses

Q9. What shape is the bulge in the edge-on galaxy? 3 responses

Q10. How tightly wound are the spiral arms? 3 responses

Q11. How many spiral arms are there? 6 responses

## Paths and the decision tree

Each galaxy's classification is the result of a specific path down a decision tree. Multiple individuals (typically 40-50) all classified the same galaxy, resulting in multiple paths along the decision tree. These multiple paths generate probabilities for each node. Volunteers begin with general questions (eg, is it smooth?) and move on to more specific ones (eg, how many spiral arms are there?).

As a result, at each node or question, the total initial probability of a classification will sum to 1.0. Those initial probabilities are then weighted as follows.

## Weighting the responses

The values of the morphology categories in the solution file are computed as follows. For the first set of responses (smooth, features/disk, star/artifact), the values in each category are simply the likelihood of the galaxy falling in each category. These values sum to 1.0. For each subsequent question, the probabilities are first computed (these will sum to 1.0) and then multiplied by the value which led to that new set of responses. 

Here is a simplified example: a galaxy had 80% of users identify it as smooth, 15% as having features/disk, and 5% as a star/artifact.

Class1.1 = 0.80

Class1.2 = 0.15

Class1.3 = 0.05

For the 80% of users that identified the galaxy as "smooth", they also recorded responses for the galaxy's relative roundness. These votes were for 50% completely round, 25% in-between, and 25% cigar-shaped. The values in the solution file are thus:

Class 7.1 = 0.80 * 0.50 = 0.40

Class 7.2 = 0.80 * 0.25 = 0.20

Class 7.3 = 0.80 * 0.25 = 0.20

This method of cumulatively multiplying probabilities applies for every morphology class, as mapped by the figure above. The sum of Class 1.1-1.3 each galaxy will always sum to 1.0, since this questions are answered for every galaxy. Class 6.1 and 6.2 have also been normalized to sum to 1.0, removing the effect of choosing 1.3 (star/artifact). For the remaining classes, the responses will always sum to <= 1.0.

The reason for this weighting is to emphasize that a good solution must at a minimum get the high-level, large-scale morphology categories correct. The best solutions, though, will also have high levels of accuracy on the detailed solutions that are further down the decision tree.

##Previous results

The Galaxy Zoo 2 project was described in a paper by Willett et al. (2013), MNRAS, 435, 2835. Contestants are welcome to read the paper, but are cautioned that use of any external data sets (including those in this paper) are strictly forbidden by the contest rules.

As a possible benchmark, we also point out a recent paper from the astronomical literature. Banerji et al. (2010), MNRAS, 406, 342 were able to distinguish smooth galaxies from feature/disks at greater than 90% accuracy. This corresponds to Class1.1 - 1.3 in this data set, and a good solution should be able to at least match that for this challenge. The expected challenge will be to get accurate predictions for the remaining 34 categories, most of which center on smaller structures in the image.

This contest is centered on image analysis; given the JPG files used by the volunteers, analyze them and see how well you can reproduce their classifications in the various classes.