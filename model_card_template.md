# Model Card

## Model Details
This model is a binary classification mode that is trained using Logistic Regression from scikit-learn. It predicts whether a person earns more than $50K/yer based on census and demographic data.

## Intended Use
This model is inteded for educational purposes to demonstrate machine learning deployment workflows. It showcases how to build, test, and deploye a machine learning system.

## Training Data
The training data is based on the UCI Adult Census Income dataset.  
It includes demographic and employment-related information like age, education, occupation, hours worked per week, and country of origin.

## Evaluation Data
The evaluation data is a held-out portion of the original census dataset created using a train-test split. The data was not seen by the model during training.

## Metrics
The following metrics were used:
- Precision
- Recall
- F1 Score

On the test dataset, the model achieved the following:
- Prcision: 0.72
- Recall: 0.58
- F1 Score: 0.64

## Ethical Considerations
The model uses sensitive demographics like race, sex, and marital status. Using this model for real-world decisions could lead to unfair or discriminatory outcomes.

## Caveats and Recommendations
This model was trained on data that is considered historical and may not generalize well to current data. 

Future Improvements:
- Using more advanced models
- Collecting more recent and diverse data
- Perform deeper fairnesn analysis