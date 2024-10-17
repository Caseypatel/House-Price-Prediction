# House Price Prediction
## Context:
- The objective of this project is to develop a **Machine Learning model** capable of predicting house sale prices based on various relevant features. 
- The dataset comprises **79 explanatory variables** that provide comprehensive insights into nearly every aspect of residential properties in Ames, Iowa.

## Background:
- In today's dynamic real estate market, the significance of accurate house price predictions is escalating rapidly. These predictions have the potential to empower homeowners, buyers, and real estate professionals alike by providing valuable insights into property values.
- The accuracy of house price predictions is crucial in the real estate industry, as it directly impacts the buying and selling decisions of stakeholders. 

## Actions:
In this project I developed a Function to Train Model using Different Regression Algorithms. Then, applied different alogrothims to the datasets. Here are the key steps:

1) Performed EDA on target variable, continuous numeric features, categorical features to determine the skewness.
2) Performed Feature Engineering for the robustness of the model.
3) Next Step was to perform LabelEncoding, One Hot Coding and Scaled data using RobustScaler.
4) Created a function to Train Model using Different Regression Algorithms.
5) Finally, created a model and based on results applied hyper parameter tunning and then staked a model.
6) Predicted test data using stacked model rewrite this in professsional way and in first person

## Results:
- The best performing models were **CatBoostRegressor, GradientBoostingRegressor & LGBMRegressor.**
- Applied **hyper-parameter tuning** to the models and stacked those models to create a more robust model.
  - The model demonstrated a strong correlation (R2 Score) of 0.876 between predicted and actual house prices.
  - The RMSE Score of 0.137 indicated a low average error in the model's predictions.
- And predicted test data. 

# EDA:   
1). Analyzed & visualized **target variable (SalePrice)**
      <p style="margin-top:15px;">
           <img src="https://github.com/user-attachments/assets/dd8e78f5-68a9-48db-9448-32f5e11d174a">
      </p>
**Results**:
- The target feature is **right-skewed distribution** due to positive Outliers.
- To achieve a Normal Distribution I used different transformation techniques like: Johnsonsu Transformation, Norm Transformation or Log Normal Transformation
        <p style="margin-top:15px;">
            <img src="https://github.com/user-attachments/assets/bbd9a80e-2f24-42d2-a64f-7a855f940da2">
        </p>
     - After applying different transformation techniques the best result were given by Unbounded Johnson Transformation.

2). Then, I visualized the Skewness of **Continous Numerical Features**
      <p style="margin-top:15px;">
          <img src="https://github.com/user-attachments/assets/e2ed33ff-4c51-4794-9356-9239023f3dd8">
      </p>
**Results**:
- Features like **3SsnPorch,LowQualFinSF,LotArea,PoolArea and MiscVal** were having extremly high skewness which can create model-complexity.
    
3). I visualized **Correlation of Continous Numerical Features** w.r.t SalesPrice.
      <p style="margin-top:15px;">
           <img src="https://github.com/user-attachments/assets/8a2e0bc9-004d-4b11-aded-e2c636354028">
      </p>
**Results:**
- Features like **1stFlrSF,GrLivArea,and GarageArea** were having **strong relation** with the target variable.
- Features like **WoodDeckSF,LotDrontage,and MasVnrArea** were having **modearte relation** with the target varible.
- Features like **LowQualFinSF,MiscVal,BsmtFinSF2,PoolArea,3SsnPorch,and ScreenPorch** were having very **low relation** with the target variable.

# Feature Engieering 

1). Created Two New Features **"RenovationStatus" and "AgeAtSale"** of the House and visualized with avg sales price
       <p style="margin-top:15px;">
        <img src ="https://github.com/user-attachments/assets/7c2854db-a959-4783-bc31-d47a7e046859">
       </p>
**Results:**
- The SalePrice for both the RenovationStatus cateegory is approxiamately same.
- There is a negative correlation between **AgeAtScale & SalePrice.** So this new feature seems very useful for model training.

2). Creating a New Feature "TotalBathrooms" using all the columns storing "Bathroom Values".
       <p style="margin-top:15px;">
        <img src="https://github.com/user-attachments/assets/ed28ebc7-cd00-4e80-887d-8362781b8176">
       </p>
**Results:**
- We can clearly observe a strong positive correlation between **Total Bathrooms and SalePrice.**
        
3). Next, I created **Total_Porch_SF** using all the columns related to "porch".
        <p style="margin-top:15px;">
             <img src="https://github.com/user-attachments/assets/0effc8f8-a548-41cf-87df-802f97365934">
        </p>
**Results:**
- Feature like **OpenPorchSF,WoodDeckSF and Total_Porch_SF** were having moderate correlation. Those features were useful.
- Feature like **3SsnPorch,EnclosedPorch, and ScreenPorch** were having weak correlation. And dropped those features.

4). Created a new feature **Total_sqr_footage** by adding all  "Sqaure Footage" variables.
       <p style="margin-top:15px;">
             <img src="https://github.com/user-attachments/assets/3bdaa903-0057-4536-bac5-422b6c62cf99">
       </p>
**Results:**
- The new feature **Total_sqr_footage and 1stFlrSF** were having very **high correlation** with the target varibale.
- Features like **BsmtFinSF1,TotalBsmtSF, and 2ndFlrSF** were having **modearte correlation** with the target variable.
- Features like **BsmtFinSF2 and BsmtUnfSF** were having very weak correaltion witht the target variable and **dropped those featues.**

5). Created a New Feature **"condition"** using **"Condition1" & "Condition2".**
   ```
    def condition(df):
    df["Condition2"] = df["Condition2"].replace({"Norm":""}) #Norm means normal which indicates there's no second condition
    combined_condition = []
    for val1,val2 in zip(df["Condition1"],df["Condition2"]):
        if val2 == "":
            combined_condition.append(val1)
        elif val1==val2:
            combined_condition.append(val1)
        else:
            combined_condition.append(val1+val2)
            
    df["Combined_Condition"] = combined_condition
    df["ProximityStatus"] = (df["Combined_Condition"] == "Norm").astype(int)
   ```

6). Created a new feature **"Heating"** using the other features such as **"Heating" and "HeatingQC"**
```
train_df.drop(columns=["Condition1","Condition2","Combined_Condition"],inplace=True)
test_df.drop(columns=["Condition1","Condition2","Combined_Condition"],inplace=True)
```

7). Lastly, created **boolean features**
       <p style="margin-top:15px;">
        <img src="https://github.com/user-attachments/assets/679ef468-2098-4220-82ee-dc88f90caf29">
       </p>
**Results:**
- All these features seemed very useful for model training.

# Data Processing 

1). Performed **Log Transformation** on target variable.

![image](https://github.com/user-attachments/assets/a21af503-2716-4a95-9706-8156c6c89bac)
**Results:**
- We can clearly observe that SalePrice has been transformed to a **normal distribution.** 

2) Then, applied Box-Cox Transformation on Continous Numerical Features to Reduce Skewness.
```
train_df[con_cols].skew().sort_values().to_frame().rename(columns={0:"Skewness"}).T
```

3). Performed **Target Encoding** on Categorical Features with High Cardinality.
```
cols = ["Neighborhood","Exterior1st","Exterior2nd","HeatingQuality"]
for column in cols:
    data = train_df.groupby(column)["SalePrice"].mean()
    for value in data.index:
        train_df[column] = train_df[column].replace({value:data[value]})
        test_df[column] = test_df[column].replace({value:data[value]})
```

4). Performed **Encoding** on Other Features.
```
encoder = LabelEncoder()

train_df[cols] = train_df[cols].apply(encoder.fit_transform)
test_df[cols] = test_df[cols].apply(encoder.fit_transform)
```

5). Applied **One-Hot Encoding** on Nominal Categorical Columns.
```
cols = train_df.select_dtypes(include="object").columns
train_df = pd.get_dummies(train_df, columns=cols)
test_df = pd.get_dummies(test_df,columns=cols)
```

6). Lastly, Scaled using **RobustScaler**
```
scaler =RobustScaler()
X_scaled = scaler.fit_transform(X)
test_df = scaler.fit_transform(test_df)
```

# Model creation & evalution
- Created a **Function** to Train Model using Different Regression Algorithms.
  -  Linear Regression Model
  -  Support vector Regressor Model
  -  Random Forest Regressor Model
  -  AdaBoost Regressor Model
  -  Gradient Boosting Regressor Model
  -  LGBM Regressor Model
  -  XGBRegressor Model
  -  CatBoost Regressor Model

- Then, compared the model's performances
![image](https://github.com/user-attachments/assets/58fcaf7f-2cd0-44e9-8a1e-c6cd7981d109)

**Result:**
- Applied hyper-parameter tunning to the best performing model **CatBoostRegressor, GradientBoostingRegressor & LGBMRegressor models**

# Lastly, stacked three models for the best outcome
```
stack_model = StackingCVRegressor(regressors=(catboost_model,gradient_model,lgbm_model),
                                  meta_regressor = catboost_model,
                                  use_features_in_secondary=True)
```
**Result:**
- The model demonstrated a strong correlation (R2 Score) of 0.876 between predicted and actual house prices.
- The RMSE Score of 0.137 indicated a low average error in the model's predictions.
- The Adjusted R2 Score of 0.756 accounted for the number of predictors in the model, providing a reliable measure of its performance.

# Predicted Test Datset using Stacked Model
```
test_preds = stack_model.predict(test_df)
sdf = test_id.to_frame()
sdf["SalePrice"] = np.floor(np.expm1(test_preds))
```
![image](https://github.com/user-attachments/assets/5e734d32-a2ae-466a-b0b0-70dc2d258d5c)









                                                                      

   






