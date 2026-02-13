
**Problem statement**: Breast Cancer classification using following 6 classification models.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

**Dataset description** : sklearn breast cancer dataset
Diagnostic Wisconsin Breast Cancer Database. This is a classic binary classification dataset available in the sklearn library. It contains features computed from a digitized image of a Fine Needle Aspirate (FNA) of a breast mass, describing the characteristics of the cell nuclei present in the image.
Dataset Shape: (569, 31)

**Comparison Table with the evaluation metrics for all 6 models**

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|
|Logistic Regression|0\.9842|0\.9974|0\.9833|0\.9916|0\.9874|0\.9661|
|Decision Tree|0\.9895|0\.9887|0\.9916|0\.9916|0\.9916|0\.9774|
|kNN|0\.9736|0\.9951|0\.9672|0\.9916|0\.9793|0\.9437|
|Naive Bayes|0\.9420|0\.9895|0\.9402|0\.9692|0\.9545|0\.8754|
|Random Forest (Ensemble)|0\.9930|0\.9995|0\.9916|0\.9972|0\.9944|0\.9850|
|XGBoost (Ensemble)|0\.9912|0\.9987|0\.9916|0\.9944|0\.9930|0\.9812|

**Observations about model performance** :
|ML Model Name|Observation about model performance|
|---|---|
|Logistic Regression|Demonstrates excellent discrimination capability with a high AUC of 0.9974, proving to be a highly effective baseline model.|
|Decision Tree|Achieves a strong balance between false positives and false negatives, indicated by identical Precision and Recall scores of 0.9916.|
|kNN|Maintains high sensitivity (Recall of 0.9916) but exhibits the second-lowest precision, suggesting a slightly higher rate of false positives compared to the ensemble models.q|
|Naive Bayes|Records the lowest scores across all metrics (Accuracy of 0.9420 and MCC of 0.8754), making it the least effective model for this dataset.|
|Random Forest (Ensemble)|Outperforms all other models, achieving the highest values in Accuracy ($0.9930$), AUC ($0.9995$), and F1 score ($0.9944$).|
|XGBoost (Ensemble)|Delivers top-tier performance as a very close second to Random Forest, showing exceptional robustness with an accuracy of 0.9912.|
