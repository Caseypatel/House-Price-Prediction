# House Price Prediction
## Context:
- The objective of this project is to develop a **Machine Learning model** capable of predicting house sale prices based on various relevant features. 
- The dataset comprises **79 explanatory variables** that provide comprehensive insights into nearly every aspect of residential properties in Ames, Iowa.

## Background:
- In today's dynamic real estate market, the significance of accurate house price predictions is escalating rapidly. These predictions have the potential to empower homeowners, buyers, and real estate professionals alike by providing valuable insights into property values.
- The accuracy of house price predictions is crucial in the real estate industry, as it directly impacts the buying and selling decisions of stakeholders. 

## Actions:
In this project I developed a Function to Train Model using Different Regression Algorithms. Then, applied different alogrothims to the datasets. Here are the key steps:
- Performed EDA on target variable, continuous numeric features, categorical features to determine the skewness.
    1). Analyzed & visualized target variable (SalePrice)
           ![image](https://github.com/user-attachments/assets/dd8e78f5-68a9-48db-9448-32f5e11d174a)
    **Results**:
     - The target feature is **right-skewed distribution** due to positive Outliers.
     - To achieve a Normal Distribution I used different transformation techniques like: Johnsonsu Transformation, Norm Transformation or Log Noraml Transformation
      ![image](https://github.com/user-attachments/assets/e19e1157-3d10-4c99-96b2-16ca0a15a101)
     - After applying different transformation techniques the best result were given by Unbounded Johnson Transformation.



