# Pharmaceutical Investment Analysis

## Overview

This is a data science portfolio project that demonstrates the integration of clinical, financial, and regulatory data to predict pharmaceutical drug trial success and identify investment opportunities. The application combines real-world data from ClinicalTrials.gov with financial market data from Yahoo Finance to build predictive models using machine learning. The project showcases end-to-end data science capabilities including data acquisition, feature engineering, model training, and interactive visualization through a Streamlit web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application for interactive data visualization and analysis
- **Visualization Libraries**: 
  - Plotly for interactive charts and financial data visualization
  - Seaborn and Matplotlib for statistical plotting
  - Custom CSS styling for enhanced UI/UX
- **Layout**: Wide layout with expandable sidebar navigation and tabbed interface for different analysis sections

### Backend Architecture
- **Modular Design**: Application split into focused modules:
  - `app.py`: Main Streamlit application and UI orchestration
  - `data_acquisition.py`: External data fetching and simulation logic
  - `modeling.py`: Machine learning pipeline and model training
  - `utils.py`: Utility functions for data formatting and categorization
- **Data Processing**: Pandas-based data manipulation with NumPy for numerical computations
- **Machine Learning**: Scikit-learn pipeline with Random Forest classification for trial outcome prediction

### Data Architecture
- **Real-time Data Sources**: 
  - ClinicalTrials.gov API for clinical trial information
  - Yahoo Finance API (via yfinance) for pharmaceutical stock data
- **Simulated Data**: Rule-based outcome generation for clinical trial success/failure due to lack of publicly available outcome data
- **Feature Engineering**: Multi-dimensional feature creation including:
  - Company financial metrics (market cap categorization)
  - Trial characteristics (phase encoding, enrollment size)
  - Temporal features and historical trial counts

### Machine Learning Pipeline
- **Problem Type**: Binary classification for trial outcome prediction
- **Model Selection**: Random Forest Classifier with cross-validation
- **Feature Processing**: 
  - Label encoding for categorical variables
  - Standard scaling for numerical features
  - Missing value imputation strategies
- **Evaluation Metrics**: Classification reports, confusion matrices, ROC curves, and feature importance analysis

## External Dependencies

### APIs and Data Sources
- **ClinicalTrials.gov API**: RESTful API for accessing clinical trial registry data with filtering capabilities for drug interventions and trial statuses
- **Yahoo Finance**: Financial market data via yfinance Python library for historical stock prices and company fundamentals

### Python Libraries
- **Web Framework**: Streamlit for rapid web application development
- **Data Science Stack**: 
  - Pandas for data manipulation and analysis
  - NumPy for numerical computing
  - Scikit-learn for machine learning algorithms and preprocessing
- **Visualization**: 
  - Plotly for interactive charts and financial visualizations
  - Matplotlib and Seaborn for statistical plotting
- **Data Acquisition**: 
  - Requests for HTTP API calls
  - yfinance for financial data retrieval

### Development Tools
- **Error Handling**: Warnings suppression and exception handling for API requests
- **Data Validation**: Input validation and data quality checks throughout the pipeline
- **Performance**: Caching mechanisms for API responses and computational results
