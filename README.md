# ML_Assignment_03

Assignment treats the Car Price dataset as a classification problem.

Selling price was divided into four classes: 0, 1, 2, and 3. 
A Multinomial Logistic Regression model was implemented.

Two models were compared:
1. Logistic Regression without regularization
2. Ridge Logistic Regression with L2 regularization
Logistic Regression model without Ridge regularization achieved the better performance and was selected as the final model.

The models were evaluated using:
1. Accuracy
2. Precision
3. Recall
4.F1-score
5. Macro averaging
6. Weighted averaging

MLflow was used to track the experiments locally.

A Dash web application was created to predict the selling price class of a car 

GitHub Actions is used for automated testing and deployment.

1. Runs two unit tests whenever code is pushed to the repository.
2. Checks that the model accepts the expected input.
3. Checks that the model output has the expected shape.
4. Deploys the application to Render only if the tests pass.

