import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def prepare_features(data):
    """
    Prepare features for machine learning model
    
    Parameters:
    data (pd.DataFrame): Raw clinical trial data with outcomes
    
    Returns:
    tuple: (X, y, feature_names)
    """
    
    if 'trial_outcome' not in data.columns:
        raise ValueError("trial_outcome column not found in data")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Clean data - remove rows with NaN in target variable
    df = df.dropna(subset=['trial_outcome'])
    
    # Ensure target variable is numeric and clean
    df['trial_outcome'] = pd.to_numeric(df['trial_outcome'], errors='coerce')
    df = df.dropna(subset=['trial_outcome'])
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning target variable")
    
    # Target variable
    y = df['trial_outcome'].values
    
    # Feature engineering
    features = []
    feature_names = []
    
    # 1. Phase encoding
    if 'phase' in df.columns:
        phase_encoder = LabelEncoder()
        phase_encoded = phase_encoder.fit_transform(df['phase'].fillna('Unknown'))
        features.append(phase_encoded.reshape(-1, 1))
        feature_names.append('phase_encoded')
        
        # Phase-specific dummy variables
        phase_dummies = pd.get_dummies(df['phase'], prefix='phase').values
        features.append(phase_dummies)
        feature_names.extend([f'phase_{col}' for col in pd.get_dummies(df['phase'], prefix='phase').columns])
    
    # 2. Sponsor class encoding
    if 'sponsor_class' in df.columns:
        sponsor_class_encoder = LabelEncoder()
        sponsor_class_encoded = sponsor_class_encoder.fit_transform(df['sponsor_class'].fillna('Unknown'))
        features.append(sponsor_class_encoded.reshape(-1, 1))
        feature_names.append('sponsor_class_encoded')
    
    # 3. Enrollment size (log transformed to handle skewness)
    if 'enrollment' in df.columns:
        enrollment_log = np.log1p(df['enrollment'].fillna(df['enrollment'].median()))
        features.append(enrollment_log.values.reshape(-1, 1))
        feature_names.append('enrollment_log')
        
        # Enrollment categories
        enrollment_categories = pd.cut(
            df['enrollment'].fillna(0), 
            bins=[0, 50, 200, 500, 1000, float('inf')], 
            labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        )
        enrollment_dummies = pd.get_dummies(enrollment_categories, prefix='enrollment_cat').values
        features.append(enrollment_dummies)
        feature_names.extend([f'enrollment_cat_{col}' for col in pd.get_dummies(enrollment_categories, prefix='enrollment_cat').columns])
    
    # 4. Study duration (if dates available)
    if 'start_date' in df.columns and 'completion_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['completion_date'] = pd.to_datetime(df['completion_date'], errors='coerce')
        
        study_duration = (df['completion_date'] - df['start_date']).dt.days
        study_duration = study_duration.fillna(study_duration.median())
        study_duration_log = np.log1p(study_duration.clip(lower=0))
        
        features.append(study_duration_log.values.reshape(-1, 1))
        feature_names.append('study_duration_log')
    
    # 5. Sponsor experience (count of previous trials)
    if 'sponsor' in df.columns:
        sponsor_counts = df['sponsor'].value_counts()
        sponsor_experience = df['sponsor'].map(sponsor_counts).fillna(1)
        sponsor_experience_log = np.log1p(sponsor_experience)
        
        features.append(sponsor_experience_log.values.reshape(-1, 1))
        feature_names.append('sponsor_experience_log')
    
    # 6. Market cap features (if available)
    if 'market_cap' in df.columns:
        market_cap = df['market_cap'].fillna(df['market_cap'].median())
        market_cap_log = np.log1p(market_cap)
        
        features.append(market_cap_log.values.reshape(-1, 1))
        feature_names.append('market_cap_log')
        
        # Market cap categories
        market_cap_categories = pd.cut(
            market_cap,
            bins=[0, 1e9, 10e9, 50e9, 100e9, float('inf')],
            labels=['Micro', 'Small', 'Mid', 'Large', 'Mega']
        )
        market_cap_dummies = pd.get_dummies(market_cap_categories, prefix='market_cap_cat').values
        features.append(market_cap_dummies)
        feature_names.extend([f'market_cap_cat_{col}' for col in pd.get_dummies(market_cap_categories, prefix='market_cap_cat').columns])
    
    # 7. Stock volatility (if available)
    if 'stock_volatility' in df.columns:
        volatility = df['stock_volatility'].fillna(df['stock_volatility'].median())
        features.append(volatility.values.reshape(-1, 1))
        feature_names.append('stock_volatility')
    
    # 8. Stock returns (if available)
    if 'stock_return_6m' in df.columns:
        returns = df['stock_return_6m'].fillna(0)
        features.append(returns.values.reshape(-1, 1))
        feature_names.append('stock_return_6m')
    
    # 9. Condition-based features
    if 'condition' in df.columns:
        # Count trials per condition
        condition_counts = df['condition'].value_counts()
        condition_frequency = df['condition'].map(condition_counts).fillna(1)
        
        features.append(np.log1p(condition_frequency).values.reshape(-1, 1))
        feature_names.append('condition_frequency_log')
        
        # Top conditions as dummies
        top_conditions = df['condition'].value_counts().head(10).index
        for condition in top_conditions:
            condition_dummy = (df['condition'] == condition).astype(int)
            features.append(condition_dummy.values.reshape(-1, 1))
            feature_names.append(f'condition_{condition.replace(" ", "_").lower()}')
    
    # 10. Year-based features
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        start_year = df['start_date'].dt.year.fillna(df['start_date'].dt.year.median())
        
        # Years since 2000 (normalize)
        years_since_2000 = start_year - 2000
        features.append(years_since_2000.values.reshape(-1, 1))
        feature_names.append('years_since_2000')
    
    # Combine all features
    if features:
        X = np.concatenate(features, axis=1)
    else:
        raise ValueError("No features could be created from the data")
    
    # Handle any remaining NaN values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    return X, y, feature_names

def train_model(X, y, feature_names):
    """
    Train a machine learning model to predict trial success
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Target variable
    feature_names (list): Names of features
    
    Returns:
    tuple: (trained_model, evaluation_metrics)
    """
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    metrics = evaluate_model(rf_model, X_test_scaled, y_test, y_pred, y_pred_proba, feature_names)
    
    # Add additional metrics
    metrics['model'] = rf_model
    metrics['scaler'] = scaler
    metrics['y_test'] = y_test
    metrics['y_pred'] = y_pred
    metrics['y_pred_proba'] = y_pred_proba
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    return rf_model, metrics

def evaluate_model(model, X_test, y_test, y_pred, y_pred_proba, feature_names):
    """
    Evaluate model performance
    
    Parameters:
    model: Trained model
    X_test: Test features
    y_test: True labels
    y_pred: Predicted labels
    y_pred_proba: Prediction probabilities
    feature_names: Names of features
    
    Returns:
    dict: Evaluation metrics
    """
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        feature_importance = np.zeros(len(feature_names))
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'classification_report': class_report
    }
    
    return metrics

def predict_trial_success(model, scaler, new_data, feature_names):
    """
    Predict success probability for new trials
    
    Parameters:
    model: Trained model
    scaler: Fitted scaler
    new_data: New trial data
    feature_names: Feature names used in training
    
    Returns:
    np.array: Success probabilities
    """
    
    # Prepare features for new data
    X_new, _, _ = prepare_features(new_data)
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    success_proba = model.predict_proba(X_new_scaled)[:, 1]
    
    return success_proba

def feature_importance_analysis(model, feature_names, top_n=20):
    """
    Analyze and return top feature importances
    
    Parameters:
    model: Trained model
    feature_names: Names of features
    top_n: Number of top features to return
    
    Returns:
    pd.DataFrame: Feature importance ranking
    """
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    else:
        return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})

def model_interpretation(model, X_test, feature_names, sample_size=100):
    """
    Provide model interpretation insights
    
    Parameters:
    model: Trained model
    X_test: Test features
    feature_names: Names of features
    sample_size: Number of samples for interpretation
    
    Returns:
    dict: Interpretation insights
    """
    
    # Sample data for interpretation
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_indices]
    else:
        X_sample = X_test
    
    # Feature importance
    feature_importance = feature_importance_analysis(model, feature_names, top_n=10)
    
    # Feature statistics
    feature_stats = pd.DataFrame({
        'feature': feature_names,
        'mean': np.mean(X_sample, axis=0),
        'std': np.std(X_sample, axis=0),
        'min': np.min(X_sample, axis=0),
        'max': np.max(X_sample, axis=0)
    })
    
    interpretation = {
        'feature_importance': feature_importance,
        'feature_statistics': feature_stats,
        'model_type': type(model).__name__,
        'sample_size': len(X_sample)
    }
    
    return interpretation
