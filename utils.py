import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def format_currency(value, currency='USD', millions=True):
    """
    Format currency values for display
    
    Parameters:
    value (float): Numeric value
    currency (str): Currency code
    millions (bool): Whether to show in millions
    
    Returns:
    str: Formatted currency string
    """
    if pd.isna(value) or value == 0:
        return f"${0:.1f}M" if millions else "$0"
    
    if millions:
        return f"${value/1e6:.1f}M"
    else:
        if value >= 1e9:
            return f"${value/1e9:.1f}B"
        elif value >= 1e6:
            return f"${value/1e6:.1f}M"
        elif value >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.0f}"

def calculate_market_cap_category(market_cap):
    """
    Categorize companies by market capitalization
    
    Parameters:
    market_cap (float): Market capitalization in USD
    
    Returns:
    str: Market cap category
    """
    if pd.isna(market_cap) or market_cap == 0:
        return "Unknown"
    
    if market_cap < 2e9:
        return "Small Cap"
    elif market_cap < 10e9:
        return "Mid Cap"
    elif market_cap < 200e9:
        return "Large Cap"
    else:
        return "Mega Cap"

def clean_sponsor_name(sponsor_name):
    """
    Clean and standardize sponsor names
    
    Parameters:
    sponsor_name (str): Raw sponsor name
    
    Returns:
    str: Cleaned sponsor name
    """
    if pd.isna(sponsor_name):
        return "Unknown"
    
    # Common cleaning patterns
    sponsor_name = str(sponsor_name).strip()
    
    # Remove common suffixes
    suffixes_to_remove = [
        ', Inc.', ', Inc', ', LLC', ', Ltd.', ', Ltd', 
        ', Corp.', ', Corp', ', Company', ', Co.',
        ' Incorporated', ' Corporation', ' Limited'
    ]
    
    for suffix in suffixes_to_remove:
        if sponsor_name.endswith(suffix):
            sponsor_name = sponsor_name[:-len(suffix)]
    
    # Standardize common pharmaceutical company names
    name_mapping = {
        'Pfizer Inc': 'Pfizer',
        'Johnson & Johnson': 'Johnson & Johnson',
        'Merck Sharp & Dohme': 'Merck',
        'Novartis Pharmaceuticals': 'Novartis',
        'Roche/Genentech': 'Roche',
        'AstraZeneca': 'AstraZeneca',
        'Bristol-Myers Squibb': 'Bristol-Myers Squibb',
        'AbbVie': 'AbbVie',
        'Eli Lilly and Company': 'Eli Lilly',
        'Gilead Sciences': 'Gilead Sciences'
    }
    
    for key, value in name_mapping.items():
        if key.lower() in sponsor_name.lower():
            return value
    
    return sponsor_name

def calculate_trial_risk_score(trial_data):
    """
    Calculate a risk score for clinical trials
    
    Parameters:
    trial_data (pd.Series): Single trial data
    
    Returns:
    float: Risk score (0-1, where 1 is highest risk)
    """
    risk_score = 0.5  # Base risk
    
    # Phase-based risk
    phase_risk = {
        'Early Phase 1': 0.8,
        'Phase 1': 0.7,
        'Phase 1/2': 0.6,
        'Phase 2': 0.65,
        'Phase 2/3': 0.55,
        'Phase 3': 0.35,
        'Phase 4': 0.2,
        'Not Applicable': 0.7,
        'Unknown': 0.8
    }
    
    if 'phase' in trial_data:
        risk_score = phase_risk.get(trial_data['phase'], 0.7)
    
    # Sponsor class adjustment
    if 'sponsor_class' in trial_data:
        if trial_data['sponsor_class'] == 'INDUSTRY':
            risk_score *= 0.8  # Industry sponsors typically lower risk
        elif trial_data['sponsor_class'] in ['OTHER_GOV', 'FED']:
            risk_score *= 1.1  # Government sponsors slightly higher risk
        else:
            risk_score *= 1.2  # Other sponsors higher risk
    
    # Enrollment size adjustment
    if 'enrollment' in trial_data:
        enrollment = trial_data['enrollment']
        if enrollment >= 1000:
            risk_score *= 0.8  # Large trials lower risk
        elif enrollment >= 500:
            risk_score *= 0.9
        elif enrollment < 50:
            risk_score *= 1.3  # Very small trials higher risk
    
    # Market cap adjustment (if available)
    if 'market_cap' in trial_data and not pd.isna(trial_data['market_cap']):
        market_cap = trial_data['market_cap']
        if market_cap > 50e9:  # Large pharma
            risk_score *= 0.7
        elif market_cap > 10e9:  # Mid-size pharma
            risk_score *= 0.8
        else:  # Small pharma
            risk_score *= 1.2
    
    # Ensure risk score stays within bounds
    return max(0.05, min(risk_score, 0.95))

def generate_investment_recommendation(trial_data, success_probability, risk_tolerance=0.5):
    """
    Generate investment recommendation based on trial data and predictions
    
    Parameters:
    trial_data (pd.Series): Trial information
    success_probability (float): Predicted success probability
    risk_tolerance (float): Investor risk tolerance (0-1)
    
    Returns:
    dict: Investment recommendation
    """
    
    risk_score = calculate_trial_risk_score(trial_data)
    
    # Investment decision logic
    if success_probability > 0.8 and risk_score < 0.3:
        recommendation = "Strong Buy"
        confidence = "High"
    elif success_probability > 0.6 and risk_score < 0.5:
        recommendation = "Buy"
        confidence = "Medium-High"
    elif success_probability > 0.4 and risk_score < risk_tolerance:
        recommendation = "Hold/Consider"
        confidence = "Medium"
    elif success_probability > 0.2:
        recommendation = "Cautious"
        confidence = "Low-Medium"
    else:
        recommendation = "Avoid"
        confidence = "High"
    
    # Calculate expected value (simplified)
    potential_revenue = 1000  # Million USD (simplified assumption)
    estimated_cost = get_phase_cost(trial_data.get('phase', 'Unknown'))
    expected_value = (success_probability * potential_revenue) - estimated_cost
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'success_probability': success_probability,
        'risk_score': risk_score,
        'expected_value': expected_value,
        'estimated_cost': estimated_cost,
        'potential_revenue': potential_revenue
    }

def get_phase_cost(phase):
    """
    Get estimated cost by trial phase (in millions USD)
    
    Parameters:
    phase (str): Trial phase
    
    Returns:
    float: Estimated cost in millions USD
    """
    
    phase_costs = {
        'Early Phase 1': 10.0,
        'Phase 1': 15.6,
        'Phase 1/2': 31.0,
        'Phase 2': 46.4,
        'Phase 2/3': 150.9,
        'Phase 3': 255.4,
        'Phase 4': 20.0,
        'Not Applicable': 25.0,
        'Unknown': 50.0
    }
    
    return phase_costs.get(phase, 50.0)

def calculate_portfolio_metrics(trials_data, weights=None):
    """
    Calculate portfolio-level metrics for a collection of trials
    
    Parameters:
    trials_data (pd.DataFrame): Multiple trials data
    weights (list): Portfolio weights (if None, equal weights assumed)
    
    Returns:
    dict: Portfolio metrics
    """
    
    if weights is None:
        weights = [1/len(trials_data)] * len(trials_data)
    
    if len(weights) != len(trials_data):
        raise ValueError("Weights length must match trials data length")
    
    # Calculate weighted metrics
    if 'success_probability' in trials_data.columns:
        portfolio_success_prob = sum(w * p for w, p in zip(weights, trials_data['success_probability']))
    else:
        portfolio_success_prob = 0.5  # Default assumption
    
    # Risk calculations
    risk_scores = [calculate_trial_risk_score(row) for _, row in trials_data.iterrows()]
    portfolio_risk = sum(w * r for w, r in zip(weights, risk_scores))
    
    # Cost calculations
    costs = [get_phase_cost(row.get('phase', 'Unknown')) for _, row in trials_data.iterrows()]
    total_cost = sum(w * c for w, c in zip(weights, costs))
    
    # Expected value
    potential_revenues = [1000] * len(trials_data)  # Simplified assumption
    expected_revenues = [p * r for p, r in zip(trials_data.get('success_probability', [0.5]*len(trials_data)), potential_revenues)]
    portfolio_expected_value = sum(w * (er - c) for w, er, c in zip(weights, expected_revenues, costs))
    
    return {
        'portfolio_success_probability': portfolio_success_prob,
        'portfolio_risk_score': portfolio_risk,
        'total_investment': total_cost,
        'expected_portfolio_value': portfolio_expected_value,
        'expected_roi': portfolio_expected_value / total_cost if total_cost > 0 else 0,
        'number_of_trials': len(trials_data),
        'risk_adjusted_return': portfolio_expected_value / (portfolio_risk + 0.01)  # Add small constant to avoid division by zero
    }

def validate_data_quality(data, required_columns=None):
    """
    Validate data quality and completeness
    
    Parameters:
    data (pd.DataFrame): Data to validate
    required_columns (list): Required columns
    
    Returns:
    dict: Data quality report
    """
    
    if required_columns is None:
        required_columns = ['nct_id', 'sponsor', 'phase', 'enrollment']
    
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_columns': [],
        'missing_data_by_column': {},
        'data_quality_score': 0.0,
        'recommendations': []
    }
    
    # Check required columns
    for col in required_columns:
        if col not in data.columns:
            quality_report['missing_columns'].append(col)
    
    # Check missing data
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_pct = missing_count / len(data) * 100
        quality_report['missing_data_by_column'][col] = {
            'missing_count': missing_count,
            'missing_percentage': missing_pct
        }
    
    # Calculate overall quality score
    avg_completeness = sum(
        (1 - info['missing_percentage']/100) 
        for info in quality_report['missing_data_by_column'].values()
    ) / len(data.columns)
    
    required_columns_present = 1 - (len(quality_report['missing_columns']) / len(required_columns))
    
    quality_report['data_quality_score'] = (avg_completeness + required_columns_present) / 2
    
    # Generate recommendations
    if quality_report['data_quality_score'] < 0.7:
        quality_report['recommendations'].append("Data quality is below recommended threshold")
    
    for col, info in quality_report['missing_data_by_column'].items():
        if info['missing_percentage'] > 20:
            quality_report['recommendations'].append(f"High missing data in column '{col}' ({info['missing_percentage']:.1f}%)")
    
    if quality_report['missing_columns']:
        quality_report['recommendations'].append(f"Missing required columns: {', '.join(quality_report['missing_columns'])}")
    
    return quality_report

def export_results(data, filename_prefix="pharma_analysis"):
    """
    Export analysis results to various formats
    
    Parameters:
    data (pd.DataFrame): Data to export
    filename_prefix (str): Prefix for filename
    
    Returns:
    dict: Export status and filenames
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_status = {
        'csv_file': None,
        'excel_file': None,
        'json_file': None,
        'success': False,
        'error_message': None
    }
    
    try:
        # CSV export
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        data.to_csv(csv_filename, index=False)
        export_status['csv_file'] = csv_filename
        
        # Excel export (if openpyxl is available)
        try:
            excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
            data.to_excel(excel_filename, index=False)
            export_status['excel_file'] = excel_filename
        except ImportError:
            pass  # openpyxl not available
        
        # JSON export
        json_filename = f"{filename_prefix}_{timestamp}.json"
        data.to_json(json_filename, orient='records', indent=2)
        export_status['json_file'] = json_filename
        
        export_status['success'] = True
        
    except Exception as e:
        export_status['error_message'] = str(e)
    
    return export_status
