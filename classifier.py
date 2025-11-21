import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from basic_info import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import sys


def get_rates(y_true, predictions):
    classes = classes = ['Negative', 'Neutral', 'Positive']
    results = {}
    
    for cls in classes:
        TP = np.sum((y_true == cls) & (predictions == cls))
        TN = np.sum((y_true != cls) & (predictions != cls))
        FP = np.sum((y_true != cls) & (predictions == cls))
        FN = np.sum((y_true == cls) & (predictions != cls))
        P = TP + FN
        N = TN + FP
        
        # Calculate derived metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN) 
        sensitivity = TP / P
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = TN / N
        
        results[cls] = {
            'P' : P,
            'N' : N,
            'TP': TP, 
            'TN': TN, 
            'FP': FP, 
            'FN': FN,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-Score': f1,
            'sensitivity' : sensitivity,
            'specificity' : specificity
        }
    
    return results

def explain_rates(results):
    output = ''
    output += f"\nRates:\n\n"
    for cls, data in results.items():
        output += f"{cls}\n" + '-' * 30 + '\n'
        for rate, value in data.items():
            output += f"{rate}:\t{value:.4f}\n"
        output += '\n'

    return output

def CM(y_test, predictions, model):
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    cm_df = pd.DataFrame(cm, 
                        index=[f'Actual_{cls}' for cls in model.classes_], 
                        columns=[f'Predicted_{cls}' for cls in model.classes_])
    return cm_df

def Data_to_File(name, model, y, X_train, X_test, y_train, y_test, label_encoders):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    with open(f"model_info/{name}.txt", "w") as f:
        accuracy = accuracy_score(y_test, predictions)
        f.write(f"Accuracy:\t{accuracy}\n\n")

        cm = CM(y_test, predictions, model)
        f.write(f'Confusion Matrix\n\n{cm}\n')

        rates = get_rates(y_test, predictions)
        f.write(f"\n{explain_rates(rates)}")

        f.write(f"\nClassification Report\n{classification_report(y_test, predictions)}")

        f.write(f"\nFeature Importance")
        f.write(f"\n{'='*50}\n")

        feature_names = X.columns
        importance_data = []
        
        # Get importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            for feature, imp in zip(feature_names, model.feature_importances_):
                importance_data.append((feature, imp, 'Gini Importance'))
        
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.classes_) == 2:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.mean(np.abs(model.coef_), axis=0)
            
            for feature, imp in zip(feature_names, importances):
                importance_data.append((feature, imp, 'Coefficient Magnitude'))
        
        else:
            # Models without built-in importance
            f.write("This model type does not provide built-in feature importance.\n")
            return
        
        # Sort and display top 10 features
        importance_data.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"\n{'Feature':25} {'ImportanTargetce':12} {'Type':20}\n")
        f.write('-' * 60 + '\n')
        
        for feature, imp, imp_type in importance_data:
            f.write(f"{feature:25} {imp:.6f}     {imp_type:20}\n")
        
        # Add correlation as reference
        f.write(f"\nCorrelation with Target:\n")
        y_encoded = LabelEncoder().fit_transform(y)
        correlations = X.corrwith(pd.Series(y_encoded)).abs().sort_values(ascending=False)
        
        for feature, corr in correlations.items():
            f.write(f"{feature:25} {corr:.6f}\n")

def prediction_table(name, model, X_test, y_test, predictions):
    if name != 'SVM':
        X_test_scaled = scaler.transform(X_test)
        prediction_probabilities = model.predict_proba(X_test_scaled)
        results_df = pd.DataFrame({
            'Actual_User_Experience': y_test.values,
            'Predicted_User_Experience': predictions,
            'Correct': (y_test.values == predictions)
        }, index=X_test.index)

        for i, class_name in enumerate(model.classes_):
            results_df[f'Probability_{class_name}'] = prediction_probabilities[:, i]

        filename = f'model_info/{name}.csv'
        results_df.to_csv(filename, index=True)


if __name__ == "__main__":
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50,        # 50 sequential trees
            learning_rate=0.1,      # How much each tree contributes
            max_depth=5,            
            random_state=42
        ),
        'Ada Boosting' : AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2), # depth 5 had a 0.32 accuracy
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        "Bagging" : BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),  
            n_estimators=50,        
            max_samples=0.8,        
            max_features=0.8,       
            bootstrap=True,         
            bootstrap_features=False, 
            random_state=42,
            n_jobs=-1               
        )
    }

    df = pd.read_csv('Dataset_BicycleUse.csv')
    df = fix_weekend(df)
    df = sort_types(df)

    if 'att_impact' in sys.argv:
        attributes = list(df.columns)

        attributes.remove('User_Experience')

        with open('Attribute_impact.txt', 'w') as f:
            for feature in attributes:
                if feature in df.columns:
                    table = attribute_impact(df, feature)
                    f.write(f'{feature}\n' + ('=' * 50) + f'\n{table}\n\n')

        with open('Attribute_impact_binned.txt', 'w') as f:
            for feature in attributes:
                if feature in df.columns:
                    table = attribute_impact_binned(df, feature)
                    f.write(f'{feature}\n' + ('=' * 50) + f'\n{table}\n\n')

    print(f"Target variable distribution:\n{df['User_Experience'].value_counts()}\n")

    X = df.drop('User_Experience', axis=1)  
    y = df['User_Experience']  


    # Convert categorical text to numbers so classifier understands them
    label_encoders = {}
    categorical_columns = ['Bike_Type', 'Gender', 'Occupation', 'Weather', 
                        'Day_of_Week', 'Time_of_Day', 'Purpose_of_Ride', 'Road_Condition']

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))  # Convert to string first
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Convert boolean columns to integers (1 = True / 0 = False)
    bool_columns = ['Is_Holiday', 'Is_Weekend', 'Helmet_Used']

    for col in bool_columns:
        X[col] = X[col].astype(int)
        print(f"Converted {col} to integers")

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,  
        stratify=y  
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"\nTest target distribution:\n{y_test.value_counts()}\n")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_encoded = LabelEncoder().fit_transform(y)
    correlation_matrix = X.copy()
    correlation_matrix['User_Experience'] = y_encoded
    correlations = correlation_matrix.corr()['User_Experience'].drop('User_Experience').abs()

    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, correlation) in enumerate(sorted_correlations, 1):
        print(f"{rank:2}. {name:25} {correlation:.4f} ({correlation*100:.2f}%)")

    #print(correlations)


    print("\nModel Comparison:")
    print("=" * 50)

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracies[name] = accuracy_score(y_test, predictions)
        
        #print(f"{name:25} Accuracy: {accuracies[name]:.4f}")
        Data_to_File(name, model, y,  X_train_scaled, X_test_scaled, y_train, y_test, label_encoders)
        prediction_table(name, model, X_test, y_test, predictions)

    print('\nModel Accuracies')

    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, accuracy) in enumerate(sorted_accuracies, 1):
        print(f"{rank:2}. {name:25} {accuracy:.4f} ({accuracy*100:.2f}%)")

    best_model_name, best_accuracy = sorted_accuracies[0]
    best_model = models[best_model_name]

    

