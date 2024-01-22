# Machine-Learning-Project-2

The following proposal dwells on the background of the business data that was
used to make predictions and better quality decisions. The business data is
collected from a campaign (what bank it is) via multiple approaches such as
mobile phone calls, text messages, and social media. The campaign was hosted to get
customers to subscribe for fixed deposits in the bank.

The domain knowledge of this dataset is related to a bank. The type of dataset is
classification. The label of the dataset is the column, ‘y’, which has either Yes or No,
which means subscribed or non-subscribed. The column ‘y’ is an imbalanced data set
with more No than Yes’s.

The reason for choosing this dataset is to understand the importance of features
that heavily impact customers. The features that contribute greatly to
making good predictions can be used to increase the number of subscribers. Increasing
the number of customers leads to a higher-quality dataset that can be analyzed to
make better decisions.

The data was downloaded from Kaggle in CSV format and imported into Jupyter
Notebook. The data was thoroughly viewed in terms of what each column means, what
data type each column is, and finalizing the label column. All the duplicates and the null
values from the data set were removed. The data type of the necessary columns was
changed to categorical type. The data of each categorical column was visualized
through multiple types of graphs such as pie graphs, bar graphs, and histograms. The data
of the numerical columns was changed for better visualization. For example, a logarithm
was applied to the data values of that table if the statistics of that table match a
right-skewed graph to demonstrate normal distribution.

The data was split into features and labels. The label is column y, which
contains whether the customers subscribed it or not. The label column was label encoded. A
pipeline was created to apply encoding to certain columns, apply logarithm, and replace
the outliers with the mode. The pipeline was used to run through multiple machine-learning models
such as Logistic Regression, Decision Trees, Random Forests, and XG
Boost. Random Forest gave the highest accuracy of 90 percent. Hyperparameter tuning
was done to by applying the grid search technique, and the parameters were tuned by using
the default value within a range of potential values to check if the accuracy can be
increased. However, in the case of this data, the accuracy was not increased.

Firstly, the Local Interpretable Model-agnostic Explanation was applied by
creating a pipeline to apply the column transformer, random over sampler, and random
forest model. Secondly, the X_train and y_train_enc values were fit into the pipeline to
create a new data frame consisting of indexes called X_train_prep. Similarly, for the
X_test, the preprocessor fit transform, was used to create a new data frame with indexes
called X_test_prep. Lastly, the X_train_prep and X_test_prep were used in Lime Tabular
Explainer method to present a black box model for each customer ID.

The MLP Classifier was used through a pipeline of preprocessors, random over
sampler, and the classifier itself. The data X_train and y_train_enc were fit into the
pipeline, and a classification report was printed with a lower accuracy of 80 percent.
Additionally, the sequential model was applied, which resulted in an accuracy of 50
percent. To conclude, neither of the deep learning models actually contributed to the
improvement of increasing the accuracy.

Therefore, the data was about the European Bank, which hosted a campaign to
bring their customers into subscribing to them for fixed deposits. The data was gathered
from the Kaggle website in CSV form and was loaded to visualize as a data frame
using pandas. The data was pre-processed by dropping null values and duplicates and
visualized to view categorical and numerical values. The data types were changed
when needed and split into training and testing sets. The split data was run
through multiple machine learning models such as Logistic Regression, Decision Trees,
Random Forests and XG Boost. The Random Forest model resulted in the highest
accuracy of 90 percent. Hyperparameter tuning was done on the Random Forest model
because it resulted in the highest accuracy using the grid search technique. The Local
Interpretable Model-agnostic Explanation was applied to view which features made the
customer to subscribe in or not. Two deep learning models: MLP Classifier and
Sequential were applied, which resulted in 80 percent and 50 percent accuracy,
respectively.
