# Predicting Home Improvement: Green Or Not?

*This is my independent capstone project for Data Science Immersive at Galvanize. It was a two-week project, managed under the Agile scrum framework with a program instructor as scrum master. Note: This project is currently underway; please check back for updates.*

**Goal:** Working with data from a home energy upgrades company,
use machine learning to predict a household's propensity to install an energy efficiency upgrade. This will enable the company to better target ideal candidates for home energy upgrades, thereby cutting down on the average cost of customer acquisition.

**Technologies Used:**
* [Python](https://www.python.org/) programming language.
* [pandas](https://pandas.pydata.org/) open source library for data analysis.
* [NumPy](http://www.numpy.org/) library for scientific computing in Python.
* [scikit-learn](http://scikit-learn.org/stable/index.html), a free software machine learning library for Python.
* In particular, scikit-learn's [Pipeline constructor](https://gist.github.com/amberjrivera/8c5c145516f5a2e894681e16a8095b5c).
* [matplotlib's Pyplot](https://matplotlib.org/api/pyplot_summary.html), [Seaborn](https://seaborn.pydata.org/index.html) and [Google](https://support.google.com/mymaps/answer/3024396?co=GENIE.Platform%3DDesktop&hl=en) for visualization.

## Contents
1. [The Question and Motivations](#the-question-and-motivations)
2. [Data Preparation](#data-preparation)
4. [Modeling and Results](#modeling-and-results)
5. [Future Work](#future-work)
6. [Acknowledgements](#acknowledgements)
6. [About Me](#about-me)

## The Question and Motivations
Looking at the fundamental building characteristics of a single-family residence along with some Census information, is it possible to identify those homes that will install an energy efficiency upgrade? Some examples of home energy upgrades are adding insulation, updating an old gas furnace, replacing a gas furnace with an electric heat pump, installing a new water heater, switching to an electric vehicle, or going solar (PV panels on the roof, or solar thermal for hot water).

For a homeowner, the benefit of upgrading is lower monthly utility bills. For the city or county running a home energy efficiency program to recruit households to upgrade, a main motivation is to reduce overall carbon emissions to meet climate change goals. For the company offering home energy audits and upgrade services, knowing which homes are more likely to upgrade reduces the overall cost of finding the right customer.

Currently, the company's strategy for acquiring new customers is guided by their ability to identify homes whose equipment is on its last legs (think heating and cooling, water heaters). It would be more powerful, however, to overlay that information with a household's likelihood to install a home energy upgrade, to identify the homes with the highest potential. That set of homes is what this project aims to identify.


## Data Preparation
The training dataset, provided to me by the company, includes data aggregated from publicly available sources and data simulated using [NREL's ResStock](https://www.nrel.gov/buildings/resstock.html) tool. The size of the data is 18,400 instances (homes), with 360 columns of both nominal and categorical attributes, and 18% of the data missing. The classes are imbalanced, with only 9% of the instances in the positive class (a home that has completed a home energy upgrade).

### Cleaning and Missing Values
I built a Preprocessing class to tidy up and flesh out the data before modeling. I dropped building attributes that were missing more than 75% of values, any redundant or irrelevant attributes, and any attributes or rows that would leak information about the target to the algorithm. I also dropped instances that were missing `last_sale_date` and `last_sale_price`, as I did not have the time to properly research that information to fill in the missing values. After those steps, the dataset was 17,300 x 205 with 30% numerical attributes and 70% categorical.

Given the time constraints, I handled missing values thoughtfully but bluntly. For numerical attributes, I filled with the median value of all non-null observations, except for a home's solar PV potential, which I filled with the mode (while there is likely diversity in the size, pitch and orientation of rooftops, homes are all located in the same 25 square mile geographic area). I filled all missing values in categorical attributes with their mode, except for `zillow_neighborhood`, which I filled with 'Unknown'.


### Features
There was a fair amount of collinearity in many of the numerical features, which tipped me off that I would need to do some regularization or choose an algorithm that would be robust to correlated features.

<img src = "visuals/fund_num_corr_mat.png" alt="Collinearity of subset of numerical features.">

I had what I suspected would be predictive information about homeowner permits, but it was sparse. To keep it, I created a summarizing feature for each home, `num_permits_since_purchase`. There is a lot more that could be done with this information, and I hope to extract more features from it in a future iteration of the analysis.

As a proxy for measuring whether social influence is a factor in a home's likelihood to install a home energy upgrade, I created three spatial clustering features, `num_upgrades_subdivision`, `num_upgrades_parcel`, and `num_upgrades_zip`. For each home, these features count the number of homes in the same group that have already upgraded. Subdivision has the highest resolution of the three with 785 different categories, and serves as a good proxy for Census Block Group, while zip code has only a handful of groups. Later, I looked at feature importances to see which of these levels of grouping would be most predictive.

### Class Imbalance
As mentioned, the classes were rather imbalanced with only 9% of the data in the positive class. In thinking and researching through the best way to handle this, I learned of diverging philosophies on the matter. There were two decisions to be made: method of balancing, and level of balancing. In my first pass, I chose to balance the classes to 75/25 majority/minority (has not upgraded; negative class / has upgraded; positive class).

In the first pass, I simply dropped the number of majority class instances necessary to reach 75/25 balance, which left a much smaller dataset to train on, 6564 x 206. As I iterate on the model, I'll instead try bootstrapping from the minority class to preserve more of the training data, and will lever the percentage of the minority class to see how it impacts performance.


## Modeling and Results
Before running the data through any algorithms, I split off a third of it (using sklearn's `train_test_split`) to set aside and use later to score the final model. It's important to have this "unseen" data in order to get a measure of model performance that is not inflated by overfitting the prediction to the training data. With the other two thirds, I used k-fold cross validation with four folds to compare classifiers for this binary classification problem.

### Identifying The Best Classifier
I started by running my clean, engineered, and balanced feature matrix through various classifiers, from simple Logistic Regression to the more complex multi-layer perceptron, to see which scored the best. I chose to evaluate my classifier based on its f1 score averaged across cross validation folds, seeking to balance the percentage of time the model predicts correctly, and the detection of the positive class.

Algorithms that generate a good f1 score are those that find harmony between precision and recall. Precision is what percentage of the time you predict correctly, while recall is the true positive rate: out of all the homes that really would do an upgrade, how many did you identify? I wanted to do both of these things well, but there is an inherent tradeoff between them; using f1 finds a balance. In this analysis, the benefit of correctly identifying a household that will do a home energy upgrade (true positive) outweighs the cost of interacting with a homeowner that is unlikely to upgrade (false positive). Furthermore, by not identifying a home that will upgrade (false negative), a lot of money and positive carbon impact is left on the table.

I initialized several classifiers available in sci-kit learn with their out-of-the-box defaults plus a bit of tuning based on intuition. All of the classifiers except Naive Bayes returned cross-validated f1 scores above 0.60, while logistic regression, decision tree, random forest, and both gradient and adaptive boosting were closer to 0.70.

*[visual TK: table(s) of f1 scores across classifiers]*

With a bit more manual tuning on those five classifiers, and a reasonable amount of grid searching first using sklearn's `GridSearchCV` and then `RandomizedSearchCV`, random forest and gradient boosting rose to the top of the pack. This checked out against my earlier exploratory data analysis, when I found multicollinearity among many features -- both algorithms are robust to this problem. While there are certainly datasets with a worse high dimensionality problem than this one, 200 attributes is still a lot; and, thinking about how this model may scale as the company grows, it's reasonable to think that additional attributes could make their way into the dataset. Again, both algorithms are robust to high dimensionality. So, how to choose?

Once I had my code well-structured using sklearn's `Pipeline` constructor, I decided to do a deeper grid search, pitting Random Forest against Gradient Boosting. (This part of the modeling is in progress -- I'm still researching and grid searching to understand which of these algorithms is the better solution. Furthermore, I'm validating different approaches to upstream transforms to see how each algorithm responds. So far, they are performing about the same, with the score and features used in the prediction for Gradient Boosting being slightly more stable than Random Forest).

*[visual TK: Feature importances. The overlap in agreement by the top two classifiers]*


### Results
*[table TK: hyper parameters, f1 score and final top 10 features after tuning final algorithm]*

[TK: Description of final algorithm; final feature importances]

One of the benefits of using `Pipeline` and `GridSearchCV` together is that you can more conveniently treat data preparation steps as hyperparameters. This allows you to tune not only the final algorithm, but the entire process. For instance, you can do blunt imputations to fill missing values to get a working model; tune that model to reasonable performance; then experiment with swapping out more complex methods for the cleaning and engineering transforms upstream of the model to see if performance improves.

I did this a bit by swapping in different methods for dimensionality reduction/feature selection, but that became less important when I selected the [Random Forest/Gradient Boosting] algorithm, which already does a good job of handling highly correlated attributes and high dimension datasets. In future iterations of the analysis, I'd like to swap out various imputation techniques for missing data, to see if there is any more performance to squeeze out of the algorithm.

*[potential visual: performance comparison when undersampling the majority class vs boostrapping vs SMOTE]*


Here's a map that demonstrates the model's performance. I backtested my model on historic data for 2016 to see how my predictions compared to reality.

*[visual TK: map of 2016 hindcast]*


### Business Implications
*[visual TK: confusion matrix and profit curve]*


## Future Work
Here are some ideas for improving upon this project:
- Try and compare more sophisticated imputation methods, such as plugging into the Zillow API to fill in missing sale date and sale price when predicting on new data.
- Further engineer off of the permits features, encoding their most useful info into the dataset.
- Find and incorporate relevant behavioral information for each household to improve performance.
- Incorporate unsupervised techniques to see whether there are any natural patterns or clusters among the data.
- In production, stand up two models and compare their performance predicting on new data, ongoing. I'd be curious to see how each of the balancing techniques I tried perform in the wild -- did bootstrapping or SMOTE improve performance at the cost of introducing bias into the model, or does it still perform better than undersampling the majority class in the long run?


## Acknowledgements
- Official documentation for [pandas](https://pandas.pydata.org/pandas-docs/stable/) and [scikit-learn](http://scikit-learn.org/stable/documentation.html).
- *["Python For Data Analysis, 2nd Edition"](http://shop.oreilly.com/product/0636920050896.do)* by William McKinney (O'Reilly). Copyright 2017.
- *["Hands On Machine Learning with Scikit-Learn and TensorFlow"](http://shop.oreilly.com/product/0636920052289.do)* by Aurélien Géron (O'Reilly). Copyright 2017.
- Batista, G., Prati, R. and Monard M. [A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.7757&rep=rep1&type=pdf)
- Isaac Laughlin's [Pipeline how-to video.](https://www.youtube.com/watch?v=0UWXCAYn8rk)
- A **big** thank you to Galvanize instructor [Elliot Cohen](https://github.com/Ecohen4), who served as advisor and cheerleader on this project, and to all of my instructors during the immersive.

## About Me
I'm new to data science, but not to data analysis. My background is in modeling portfolio risk in the solar financing industry and advising on public policy based on market research and modeling. I'm available for contract data work, and seeking a position where I can program machine learning and other advanced analytics solutions in Python. I thrive in situations where people want to know the fundamental truth of their business, organization, or market as much as I do. Connect with me on [LinkedIn](www.linkedin.com/in/amberjrivera) or please [send me an email](<amberjrivera@gmail.com>) with any question, critique, or idea for this project!
