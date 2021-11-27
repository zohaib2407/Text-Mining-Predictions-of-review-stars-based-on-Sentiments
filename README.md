# Text-Mining-Predictions-of-review-stars-based-on-Sentiments
Analysis on reviews data from Yelp to predict business ratings based on review sentiments

In this analysis, we have built different predictive models using 3 different dictionaries and full matching terms. We have built models to predict review sentiment. For this, we have split the data randomly into training and test sets. To make run times manageable, we took a smaller sample of reviews.

**Using only matching dictionary terms - Comparative evaluation of different dictionaries**

For developing all our models, we decide to use the ‘tf-idf’ measure since it provides a ‘balanced’ statistical measure that considers the importance of a word in document with respect to the collection of documents. Term frequency (tf) measures how frequently a word occurs in a document but fails to capture importance of words. On the other hand, inverse document frequency (idf) is a measure that decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents but fails to give information about how important the word is to a given document. As ‘tf-idf’ measure is product of ‘tf’ an ‘idf’, it gives more weightage to the word that is rare withing the documents while providing more importance to the word that is more frequent in the document. 
We filter out all reviews that are rated ‘3 stars’ and create our dependable variable with negative class (-1) being all reviews rated ‘1 star’ or ‘2 stars’ and positive class (1) being all reviews rated ‘4 stars’ or ‘5 stars’. To keep things clear, we created four different document term matrices, one for each of three dictionaries and their combination.

**Random Forest**

To find the best model, we did a full-grid search across range of values of hyper-parameters: mtry, num.tress, 

**Model 1: Bing**
The grid search resulted into best model with parameters as mtry = 30, num.trees = 500 with ACCURACY=88%, Recall = 97% on test Dataset

![image](https://user-images.githubusercontent.com/35283246/143687714-8d605791-794e-48ca-b302-e29af08cf129.png)

**Model 2: NRC**
The grid search resulted into best model with parameters as mtry = 39, num.trees = 400 with ACCURACY = 86%, Recall = 97% on test Dataset

![image](https://user-images.githubusercontent.com/35283246/143687759-bfc03ea0-9269-408b-804a-b760571ed4ee.png)


**Model 3: AFINN**
The grid search resulted into best model with parameters as mtry = 23, num.trees = 500 with ACCURACY = 86%, Recall = 96% on test Dataset

![image](https://user-images.githubusercontent.com/35283246/143687782-47fc9757-67fc-4c4d-af9a-d16697d81451.png)

**Model 4: Combined**
The grid search resulted into best model with parameters as mtry = 49, num.trees = 500 ACCURACY = 88%, Recall = 98% on test Dataset

![image](https://user-images.githubusercontent.com/35283246/143687824-746463e8-2577-4e61-bbac-3ad50471e503.png)


Takeaway – 
We observe that random forest performs quite well in classification of sentiment, irrespective of the dictionary used. It exhibits excellent performance metric values for accuracy, precision, and recall. ROC curve also confirms that the classifier maintains a high true positive rate while also having a low false positive rate, with AUC > 90% with all models. With text classification, we are not interested how good the model classifies any class of interest. Thus, with an imbalanced class distribution and more focus on precision and recall, F-score is a better metric to compare performance.

Model	F-Score
RF Bing	0.9246
RF NRC	0.9152
RF AFINN	0.9139
RF Combined	0.9229



