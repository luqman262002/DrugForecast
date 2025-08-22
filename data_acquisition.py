import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

def fetch_clinical_trials(condition="cancer", max_studies=500):
    """
    Fetch clinical trial data from ClinicalTrials.gov API
    
    Parameters:
    condition (str): Medical condition to search for
    max_studies (int): Maximum number of studies to fetch
    
    Returns:
    pd.DataFrame: Clinical trials data
    """
    
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Parameters for the API request
    params = {
        'query.cond': condition,
        'query.intr': 'Drug',  # Focus on drug interventions
        'filter.overallStatus': 'COMPLETED|RECRUITING|ACTIVE_NOT_RECRUITING',
        'pageSize': min(max_studies, 1000),  # API limit
        'format': 'json'
    }
    
    try:
        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'studies' not in data:
            raise ValueError("No studies found in API response")
        
        studies = data['studies']
        
        # Extract relevant information
        trials_data = []
        
        for study in studies:
            try:
                # Basic study information
                nct_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', '')
                brief_title = study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', '')
                
                # Study design
                study_type = study.get('protocolSection', {}).get('designModule', {}).get('studyType', '')
                phases = study.get('protocolSection', {}).get('designModule', {}).get('phases', [])
                phase = phases[0] if phases else 'UNKNOWN'
                
                # Sponsor information
                lead_sponsor = study.get('protocolSection', {}).get('sponsorCollaboratorsModule', {}).get('leadSponsor', {})
                sponsor_name = lead_sponsor.get('name', 'Unknown')
                sponsor_class = lead_sponsor.get('class', 'Unknown')
                
                # Enrollment
                enrollment_info = study.get('protocolSection', {}).get('designModule', {}).get('enrollmentInfo', {})
                enrollment_count = enrollment_info.get('count', 0)
                
                # Status and dates
                status_module = study.get('protocolSection', {}).get('statusModule', {})
                overall_status = status_module.get('overallStatus', '')
                start_date = status_module.get('startDateStruct', {}).get('date', '')
                completion_date = status_module.get('completionDateStruct', {}).get('date', '')
                
                # Conditions
                conditions_module = study.get('protocolSection', {}).get('conditionsModule', {})
                conditions = conditions_module.get('conditions', [])
                primary_condition = conditions[0] if conditions else condition
                
                trials_data.append({
                    'nct_id': nct_id,
                    'title': brief_title,
                    'sponsor': sponsor_name,
                    'sponsor_class': sponsor_class,
                    'phase': phase,
                    'enrollment': enrollment_count,
                    'status': overall_status,
                    'start_date': start_date,
                    'completion_date': completion_date,
                    'condition': primary_condition,
                    'study_type': study_type
                })
                
            except Exception as e:
                # Skip problematic studies but continue processing
                continue
        
        df = pd.DataFrame(trials_data)
        
        # Clean and standardize data
        if len(df) > 0:
            # Standardize phase names
            phase_mapping = {
                'PHASE1': 'Phase 1',
                'PHASE2': 'Phase 2',
                'PHASE3': 'Phase 3',
                'PHASE4': 'Phase 4',
                'PHASE1|PHASE2': 'Phase 1/2',
                'PHASE2|PHASE3': 'Phase 2/3',
                'EARLY_PHASE1': 'Early Phase 1',
                'NA': 'Not Applicable',
                'UNKNOWN': 'Unknown'
            }
            df['phase'] = df['phase'].map(phase_mapping).fillna(df['phase'])
            
            # Clean enrollment data
            df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce').fillna(0).astype(int)
            
            # Filter out studies with zero enrollment
            df = df[df['enrollment'] > 0]
            
            # Limit to requested max_studies to ensure slider accuracy
            if len(df) > max_studies:
                df = df.head(max_studies)
            
            # Clean sponsor names
            df['sponsor'] = df['sponsor'].str.strip()
            
            # Convert dates
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['completion_date'] = pd.to_datetime(df['completion_date'], errors='coerce')
            
        return df
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data from ClinicalTrials.gov: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing clinical trial data: {str(e)}")

def fetch_stock_data(tickers, years_back=5):
    """
    Fetch stock data for pharmaceutical companies using yfinance
    
    Parameters:
    tickers (list): List of ticker symbols
    years_back (int): Number of years of historical data
    
    Returns:
    dict: Dictionary with ticker as key and DataFrame as value
    """
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    stock_data = {}
    
    for ticker in tickers:
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if len(hist) > 0:
                # Calculate additional metrics
                hist['Returns'] = hist['Close'].pct_change()
                hist['Volatility'] = hist['Returns'].rolling(window=30).std()
                hist['MA_50'] = hist['Close'].rolling(window=50).mean()
                hist['MA_200'] = hist['Close'].rolling(window=200).mean()
                
                # Get company info
                info = stock.info
                hist['Market_Cap'] = info.get('marketCap', np.nan)
                hist['Sector'] = info.get('sector', 'Unknown')
                
                stock_data[ticker] = hist
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
        except Exception as e:
            print(f"Warning: Could not fetch data for {ticker}: {str(e)}")
            continue
    
    return stock_data

def create_simulated_outcomes(clinical_data):
    """
    Create simulated trial outcomes based on industry statistics
    
    Parameters:
    clinical_data (pd.DataFrame): Clinical trial data
    
    Returns:
    pd.DataFrame: Data with simulated outcomes
    """
    
    # Industry success rates by phase (approximate)
    phase_success_rates = {
        'Early Phase 1': 0.40,
        'Phase 1': 0.60,
        'Phase 1/2': 0.50,
        'Phase 2': 0.35,
        'Phase 2/3': 0.45,
        'Phase 3': 0.65,
        'Phase 4': 0.85,
        'Not Applicable': 0.30,
        'Unknown': 0.30
    }
    
    # Sponsor class multipliers (industry vs academic)
    sponsor_multipliers = {
        'INDUSTRY': 1.2,
        'OTHER_GOV': 0.9,
        'FED': 0.8,
        'OTHER': 0.7,
        'Unknown': 0.8
    }
    
    # Enrollment size impact (larger trials often more successful)
    def enrollment_multiplier(enrollment):
        if enrollment >= 1000:
            return 1.3
        elif enrollment >= 500:
            return 1.2
        elif enrollment >= 100:
            return 1.1
        else:
            return 1.0
    
    outcomes_data = []
    
    for idx, row in clinical_data.iterrows():
        # Base success rate from phase
        base_rate = phase_success_rates.get(row['phase'], 0.3)
        
        # Apply sponsor multiplier
        sponsor_mult = sponsor_multipliers.get(row['sponsor_class'], 0.8)
        
        # Apply enrollment multiplier
        enrollment_mult = enrollment_multiplier(row['enrollment'])
        
        # Calculate final probability
        success_prob = min(base_rate * sponsor_mult * enrollment_mult, 0.95)
        
        # Add some randomness
        success_prob += random.uniform(-0.1, 0.1)
        success_prob = max(0.05, min(success_prob, 0.95))  # Keep within bounds
        
        # Generate binary outcome
        trial_outcome = 1 if random.random() < success_prob else 0
        
        outcomes_data.append({
            'trial_outcome': trial_outcome,
            'success_probability': success_prob
        })
    
    return pd.DataFrame(outcomes_data)

def integrate_stock_clinical_data(clinical_data, stock_data):
    """
    Integrate clinical trial data with stock market data
    
    Parameters:
    clinical_data (pd.DataFrame): Clinical trial data
    stock_data (dict): Stock price data by ticker
    
    Returns:
    pd.DataFrame: Integrated dataset
    """
    
    # Create a mapping from company names to tickers
    # This is simplified - in practice, you'd need a comprehensive mapping
    company_ticker_mapping = {
        'Pfizer': 'PFE',
        'Johnson & Johnson': 'JNJ',
        'Merck': 'MRK',
        'Novartis': 'NVS',
        'Roche': 'RHHBY',
        'AstraZeneca': 'AZN',
        'Bristol-Myers Squibb': 'BMY',
        'AbbVie': 'ABBV',
        'Eli Lilly': 'LLY',
        'Gilead': 'GILD'
    }
    
    integrated_data = clinical_data.copy()
    
    # Add stock-related features
    integrated_data['ticker'] = None
    integrated_data['market_cap'] = None
    integrated_data['stock_volatility'] = None
    integrated_data['stock_return_6m'] = None
    
    for idx, row in integrated_data.iterrows():
        sponsor = row['sponsor']
        
        # Find matching ticker
        ticker = None
        for company, tick in company_ticker_mapping.items():
            if company.lower() in sponsor.lower():
                ticker = tick
                break
        
        if ticker and ticker in stock_data:
            stock_df = stock_data[ticker]
            
            # Get market cap (latest available)
            market_cap = stock_df['Market_Cap'].iloc[-1] if not stock_df['Market_Cap'].isna().all() else None
            
            # Get volatility (average over last year)
            volatility = stock_df['Volatility'].tail(252).mean() if 'Volatility' in stock_df.columns else None
            
            # Get 6-month return
            if len(stock_df) >= 126:  # ~6 months of trading days
                return_6m = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-126]) - 1
            else:
                return_6m = None
            
            integrated_data.at[idx, 'ticker'] = ticker
            integrated_data.at[idx, 'market_cap'] = market_cap
            integrated_data.at[idx, 'stock_volatility'] = volatility
            integrated_data.at[idx, 'stock_return_6m'] = return_6m
    
    return integrated_data
