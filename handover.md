## Project Handover - Predicting Home Improvement: Green Or Not?

### Project Assets
- [ReadMe of the public Github repository](https://github.com/amberjrivera/ee-upgrades-propensity)
- Full repo, shared on Dropbox:
```
├── /data (data used in training, and where pickled model is stored)
├── /helper (value calc, helper code, guides, and research used during project)
├── /src (contains all main scripts used in analysis)
      ├── EDA_notebook.ipynb (Jupyter notebook with exploratory data analysis)
      ├── attributes.py (class built to hide attribute names in public repo)
      ├── baseline.py (quick model up and running to get a feel for signal)
      ├── model.py (compares classifiers before grid searching)
      ├── run.py (main run file)
      ├── pipeline.py (infrastructure for Sklearn's Pipeline)
      ├── search.py (calls simple Pipeline object, cross-validates and
                    grid-searches hyperparameters)
      ├── visuals.py (code to generate correlation matrix, feature importances)
├── /visuals (graphics used for github, presentation; final results)
```

### Resources
* [Data Science for Business](https://drive.google.com/file/d/0B1cm3fV8cnJwNDJFNmx2a2RBaTg/view) (In particular, chapters 7 and 8).
* [On balancing classes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.7757&rep=rep1&type=pdf)
* [On SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf)
* [The Relationship Between Precision-Recall and ROC Curves](http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
* The [Precision and recall Wikipedia page](https://en.wikipedia.org/wiki/Precision_and_recall) (recommended by our instructors for clarity and thoroughness).

### Questions
1. Collinearity matrix: Both dark red and dark blue indicate collinearity?
  - Yes, the two features that correspond to that box are either positively correlated (darker red), or negatively correlated (darker blue).

2. Tradeoff between AUC and Recall
    - There is the ROC AUC, and there is the precision-recall AUC. Since this is an unbalanced dataset, the precision-recall metric and PR-AUC are better to use than the ROC curve. Even though we balance the classes in training, the reality on unseen data is that most homes have not yet upgraded, so the distribution of data you'll predict on is still unbalanced.

    - Working to optimize the precision-recall AUC would be a good way to further this analysis. You could use [sklearn's auc](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc) for this.

3. What additional data points would you predict would lead to a higher recall score?
  - It's hard to know any particular feature's effect on a score, and you'll be constrained to the data that are available and join-able. Perhaps there are research papers in this area that have findings you could start from; otherwise, it'd be a matter of adding the data in, and testing it's utility at improving the score. You could do this either by using an algorithm that does fine with high dimensions (logistic regression with heavy L1 regularization, random forest, or boosting) and looking at feature importances, or through recursive feature elimination with the other algorithms.

  - There are some things I'd be curious to know the predictive power of for this target. For example:
    - Does the household recycle?
    - How many hours a day is the house occupied, and during which hours?
    - Is it a family, a group house, or a single occupant?
    - Is this their first home? Is it their only home? How long do they expect to live in this home?
    - How do the occupants commute?
    - Has the household ever discussed purchasing an electric vehicle?
    - What is the race, ethnicity, religion, education level of the home? (this is somewhat captured for the block in the Census data)
    - What is the dominant political affiliation of the home?

### Ideas for Further Work / Research
* See if the same score/outcomes can be replicated in Data Science Studio (after applying the same transforms, and holding out 30% of the data to use for scoring).

* Develop a custom scorer by optimizing the balance of precision and recall, based on more specific business priorities/market realities [(Sklearn's fbeta_score)](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

* Try to optimize Logistic Regression above Random Forest. After our code freeze, I learned through research that *some* have found Logistic Regression to have inherent advantages over Random Forest in the case of class imbalance. There is also overwhelming research to the contrary, so I'd be really curious to see if we could get LR to perform just as well; this dataset has surprised me in other ways. No inherent advantage to one or the other all else equal (unless you prefer one); just research.

* Take the top 10-30 features out of a fully tuned Random Forest or Gradient Boosting model (they each use different features in their predictions); drop all other features from the dataset; then start the search for the best classifier over again, to see if the algorithms that do better with fewer features can get a better Recall score on this new, narrower, dataset.

* Further feature engineer off of the permits data. E.g. estimate the time until each relevant piece of equipment will go out, for each home, and include that as a feature in the model, in addition to the num_upgrades_since_purchase feature.
