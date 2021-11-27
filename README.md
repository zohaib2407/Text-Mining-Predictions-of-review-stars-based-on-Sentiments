# Text-Mining-Predictions-of-review-stars-based-on-Sentiments
Analysis on reviews data from Yelp to predict business ratings based on review sentiments

In this analysis, we have built different predictive models using 3 different dictionaries and partial and full matching terms. We have built models to predict review sentiment. For this, we have split the data randomly into training and test sets. To make run times manageable, we took a smaller sample of reviews.

**Using only matching dictionary terms - Comparative evaluation of different dictionaries**

For developing all our models, we decide to use the ‘tf-idf’ measure since it provides a ‘balanced’ statistical measure that considers the importance of a word in document with respect to the collection of documents. Term frequency (tf) measures how frequently a word occurs in a document but fails to capture importance of words. On the other hand, inverse document frequency (idf) is a measure that decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents but fails to give information about how important the word is to a given document. As ‘tf-idf’ measure is product of ‘tf’ an ‘idf’, it gives more weightage to the word that is rare withing the documents while providing more importance to the word that is more frequent in the document. 
We filter out all reviews that are rated ‘3 stars’ and create our dependable variable with negative class (-1) being all reviews rated ‘1 star’ or ‘2 stars’ and positive class (1) being all reviews rated ‘4 stars’ or ‘5 stars’. To keep things clear, we created four different document term matrices, one for each of three dictionaries and their combination.
	Observations	Variables	Positive Class	Negative Class
      Bing	      33597	1130	25234	8363
      NRC	        34264	1558	25667	8597
      AFINN	      32902	621	24704	8198
      Combined	  34420	2103	25799	8621


