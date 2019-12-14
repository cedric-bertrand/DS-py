# DS-py
Various Python notebooks on data science problems:  
* the directory taarifa-wp contains a [notebook](https://github.com/cedric-bertrand/DS-py/blob/master/taarifa-wp/water_pumps.ipynb) for a multiclass classification problem. This is for a competition hosted by [drivendata.org](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) to predict the functional status of public water pumps in Tanzania. It is an interesting problem with several categorical features with very high cardinality (each having several thousands distinct values). I used bayesian encoding for these features and managed to rank in the top 9%.  
* an advanced [study](https://github.com/cedric-bertrand/DS-py/blob/master/titanic/titanic_pipelines.ipynb) of the Titanic dataset. I fitted 6 different estimators using ColumnTransformers, pipelines and grid search to optimize hyperparameters for thepreprocessing and predictive steps. The 6 estimators were grouped into a voting classifier to improve the predictive performance. 
* comparison of logistic regression, random forest and SVM for the Titanic data set
* implementation of a MLP and CNN on the MNISt data set 
* quick overview of 3 Python visualisation libraries (matplotlib, seaborn, bokeh)
