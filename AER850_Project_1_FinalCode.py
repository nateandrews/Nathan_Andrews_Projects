#step 1: data processing
import pandas as pd

#file path
file_path = '/Users/nathanandrews/Downloads/Project_1_Data.csv'

#load the CSV file
df = pd.read_csv(file_path)

#step 2: data visualization

#calculating stats for X, Y, and Z
step_stats_X = df.groupby('Step')['X'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'X_' + col)
step_stats_Y = df.groupby('Step')['Y'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'Y_' + col)
step_stats_Z = df.groupby('Step')['Z'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'Z_' + col)

#concatenate X, Y, and Z statistics
step_stats_XYZ = pd.concat([step_stats_X, step_stats_Y, step_stats_Z], axis=1)

#ensure all columns are displayed when printing (had issue)
pd.set_option('display.max_columns', None)

#displaying the combined dataframe
print(step_stats_XYZ)

#checking unique Y values within each step
print(df.groupby('Step')['Y'].unique())

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#converting x, y, z to numpy arrays
X = np.array(df['X'])
Y = np.array(df['Y'])
Z = np.array(df['Z'])
Step = np.array(df['Step'])

#setting up figure and 3d axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#creating colour for steps on figure
palette = sns.color_palette('tab10', n_colors=len(np.unique(Step)))

#ploting data
scatter = ax.scatter(X, Y, Z, c=Step, cmap='tab10')

#adding labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of X, Y, Z Coordinates by Step')

#adding color bar 
legend1 = ax.legend(*scatter.legend_elements(), title="Step")
ax.add_artist(legend1)

#display plot
plt.show()

#step 3: correlation analysis

#creating dataframe that contains features and target variable
data_for_corr = df[['X', 'Y', 'Z', 'Step']]

#calculating correlation maxtrix using pearson correlation
corr_matrix = data_for_corr.corr(method='pearson')

#displaying correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

#plotting correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap of X, Y, Z with Step')
plt.show()

#Step 4: classification model development
from sklearn.model_selection import train_test_split

#features and target varibale 
X = df[['X', 'Y', 'Z']]
y = df['Step']

#splitting data for 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

#scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic regression model
logistic_regression = LogisticRegression(max_iter = 1000)
logistic_params = {'C': [0.01, 0.1, 1, 10, 100]}
logistic_grid = GridSearchCV(logistic_regression, logistic_params, cv = 5)

#ensure xtrain and xtest have correct feature names for random forest
X_train_df = pd.DataFrame(X_train_scaled, columns = ['X', 'Y', 'Z'])
X_test_df = pd.DataFrame(X_test_scaled, columns = ['X', 'Y', 'Z'])

#random forest
random_forest = RandomForestClassifier()
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split':[2, 5, 10], 'min_samples_leaf':[1, 2, 4]}
rf_grid = GridSearchCV(random_forest, rf_params, cv = 5)

#support vector machine
svm = SVC()
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(estimator = svm, param_grid = svm_params, cv = 5)

#gradient boosting using RandomizedSearchCV
gradient_boosting = GradientBoostingClassifier()
gb_params = {'n_estimators': [50, 100, 200],
             'learning_rate': [0.01, 0.1, 0.2],
             'max_depth': [3, 5, 7]}
gb_random_search = RandomizedSearchCV(gradient_boosting, gb_params, cv = 5, n_iter = 10, random_state = 42)

#training the model

#fit logistic regression model
logistic_grid.fit(X_train_scaled, y_train)
print(f"Best Logistic Regression Parameters: {logistic_grid.best_params_}")

#fit random forest model
rf_grid.fit(X_train_df, y_train)
print(f"Best Random Forest Parameters: {rf_grid.best_params_}")

#fit SVM model
svm_grid.fit(X_train_scaled, y_train)
print(f"Best SVM Parameters: {svm_grid.best_params_}")

#fit gradient boosting model
gb_random_search.fit(X_train, y_train)
print(f"Best Gradient Boosting Parameters: {gb_random_search.best_params_}")

#Step 5: model performance analysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#function to evaluate and display the metrics
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted', zero_division = 0)
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    #confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6, 4))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
if 'logistic_grid' in locals() and 'rf_grid' in locals() and 'svm_grid' in locals() and 'gb_random_search' in locals():
    #evaluating models
    evaluate_model(logistic_grid.best_estimator_, X_test_scaled, y_test, "Logistic Regression")
    evaluate_model(rf_grid.best_estimator_, X_test_df, y_test, "Random Forest")
    evaluate_model(svm_grid.best_estimator_, X_test_scaled, y_test, "SVM")
    evaluate_model(gb_random_search.best_estimator_, X_test, y_test, "Gradient Boosting")
else:
    print("Make sure all models are properly trained before evaluation.")
    
    #Step 6: stacked model performance analysis
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate

    
    #defining base estimators using the best estimators from previous models
    base_estimators = [
        ('random_forest', rf_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_)
        ]
    
    #defining final estimator
    final_estimator = LogisticRegression(max_iter = 1000)
    
    #creating StackingClassifier
    stacking_clf = StackingClassifier(
        estimators = base_estimators,
        final_estimator = final_estimator,
        cv = 5
        )
    
    #fit stacking classifier on scaled training data
    print("Fitting Stacking Classifier...")
    stacking_clf.fit(X_train_scaled, y_train)
    print("Stacking Classifier trained successfully.")
    
    #evaluating stacked model
    evaluate_model(stacking_clf, X_test_scaled, y_test, "Stacked Classifier")
    
    #cross validate stacked model with metrics
    print("Running cross-validation...")
    cv_results = cross_validate(stacking_clf, X_train_scaled, y_train, cv = 5,
                                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    
    #check cross validation to ensure data is caught (had issues)
    print(f"cross-validation results: {cv_results}")
    
    #display cross validation scores for metrics
    if 'test_accuracy' in cv_results:
        print(f"Stacked Classifier Cross-Validation Accuracy: {np.mean(cv_results['test_accuracy']):.2f}")
    else:
        print("Accuracy scores not found.")
        
    if 'test_precision_weighted' in cv_results:
        print(f"Stacked Classifier Cross-Validation Precision: {np.mean(cv_results['test_precision_weighted']):.2f}")
    else:
        print("Precision scores not found.")
            
    if 'test_recall_weighted' in cv_results:
        print(f"Stacked Classifier Cross-Validation Recall: {np.mean(cv_results['test_recall_weighted']):.2f}")
    else:
        print("Recall scores not found.")
        
    if 'test_f1_weighted' in cv_results:
        print(f"Stacked Classifier Cross-Validation F1-Score: {np.mean(cv_results['test_f1_weighted']):.2f}")
    else:
        print("F1 scores not found.")
        
    #step 7:save trained stacked model and make predictions
    import joblib
   
    
    #saving model
    model_filename = 'stacked_classifier_model.joblib'
    joblib.dump(stacking_clf, model_filename)
    print(f"Model saved as {model_filename}")
    
    #loading model
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully.")
    
    #given coordinates to predict maintenance steps
    new_coordinates = np.array([[9.375, 3.0625, 1.51],
                                [6.995, 5.125, 0.3875],
                                [0, 3.0625, 1.93],
                                [9.4, 3, 1.8],
                                [9.4, 3, 1.3]])
    #scale new data using same scaler used for the training data
    new_coordinates_scaled = scaler.transform(new_coordinates)
    
    #predicting maintenance steps
    predicted_steps = loaded_model.predict(new_coordinates_scaled)
    
    #display predictions
    for i, coords in enumerate(new_coordinates):
        print(f"Coordinates: {coords} -> Predicted Maintenance Step: {predicted_steps[i]}")
        
        
        


            

    

    







