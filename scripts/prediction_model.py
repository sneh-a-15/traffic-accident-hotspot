import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_for_ml(df):
    """Preprocess data for machine learning"""
    df = df.copy()
    df = df[['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
             'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Hour']]
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)
    return df

def train_model(df, model_type='RandomForest'):
    """Train classification model"""
    X = df.drop(columns=['Severity'])
    y = df['Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, acc, X_test, y_test, preds, X.columns

def get_feature_importance(model, feature_names, top_n=10):
    """Get feature importance from trained model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    return [(feature_names[i], importances[i]) for i in indices]

def plot_confusion_matrix(y_test, preds):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_feature_importance(importances):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    features, scores = zip(*importances)
    ax.barh(range(len(features)), scores, color='steelblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    return fig
