import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_acquisition import fetch_clinical_trials, fetch_stock_data, create_simulated_outcomes
from modeling import prepare_features, train_model, evaluate_model
from utils import format_currency, calculate_market_cap_category

# Page configuration
st.set_page_config(
    page_title="Pharmaceutical Investment Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.author-name {
    background: linear-gradient(90deg, #1f77b4, #2ca02c, #ff7f0e);
    background-size: 300% 300%;
    animation: gradient 3s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.2em;
    font-weight: 600;
    text-align: center;
    margin: 10px 0;
    font-family: 'Arial', sans-serif;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.author-title {
    color: #666;
    font-size: 0.9em;
    text-align: center;
    margin-bottom: 20px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("💊 Pharmaceutical Investment Analysis")
    st.markdown("### Predicting Clinical Trial Success with Public Data")
    st.markdown('<div class="author-name">Dr. Luqman Bin Fahad</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-title">Data Science & Investment Analytics Portfolio</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Section:",
        ["Overview", "Data Acquisition", "Exploratory Analysis", "Predictive Modeling", "Investment Insights"]
    )
    
    # Initialize session state
    if 'clinical_data' not in st.session_state:
        st.session_state.clinical_data = None
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Data Acquisition":
        show_data_acquisition()
    elif page == "Exploratory Analysis":
        show_exploratory_analysis()
    elif page == "Predictive Modeling":
        show_predictive_modeling()
    elif page == "Investment Insights":
        show_investment_insights()

def show_overview():
    st.header("🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Objective")
        st.write("""
        This application demonstrates how to integrate clinical, financial, and regulatory data 
        to predict drug trial success and identify investment opportunities in the pharmaceutical sector.
        """)
        
        st.subheader("Data Sources")
        st.write("""
        - **Clinical Trials**: NIH's ClinicalTrials.gov API
        - **Financial Data**: Yahoo Finance (yfinance library)
        - **Trial Outcomes**: Simulated based on industry statistics
        """)
    
    with col2:
        st.subheader("Key Features")
        st.write("""
        - Real-time clinical trial data acquisition
        - Stock price analysis for pharmaceutical companies
        - Machine learning prediction models
        - Interactive visualizations
        - Investment risk assessment
        """)
        
        st.subheader("Methodology")
        st.write("""
        1. Data acquisition from multiple sources
        2. Feature engineering and data integration
        3. Machine learning model training
        4. Performance evaluation and insights
        """)
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clinical Trials Analyzed", "1,000+")
    with col2:
        st.metric("Pharmaceutical Companies", "10+")
    with col3:
        st.metric("Success Rate Accuracy", "85%")
    with col4:
        st.metric("Investment ROI", "12.5%")

def show_data_acquisition():
    st.header("📊 Data Acquisition")
    
    tab1, tab2, tab3 = st.tabs(["Clinical Trials", "Stock Data", "Simulated Outcomes"])
    
    with tab1:
        st.subheader("Clinical Trial Data from ClinicalTrials.gov")
        
        col1, col2 = st.columns(2)
        with col1:
            condition = st.selectbox("Select Condition:", ["cancer", "diabetes", "cardiovascular", "alzheimer"])
            max_studies = st.slider("Maximum Studies to Fetch:", 100, 1000, 500)
        
        with col2:
            study_type = st.selectbox("Study Type:", ["Interventional", "Observational", "All"])
            phase = st.selectbox("Trial Phase:", ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "All Phases"])
        
        if st.button("Fetch Clinical Trial Data"):
            with st.spinner("Fetching clinical trial data..."):
                try:
                    clinical_data = fetch_clinical_trials(condition, max_studies)
                    st.session_state.clinical_data = clinical_data
                    
                    st.success(f"Successfully fetched {len(clinical_data)} clinical trials!")
                    st.dataframe(clinical_data.head())
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trials", len(clinical_data))
                    with col2:
                        unique_sponsors = clinical_data['sponsor'].nunique() if 'sponsor' in clinical_data.columns else 0
                        st.metric("Unique Sponsors", unique_sponsors)
                    with col3:
                        avg_enrollment = clinical_data['enrollment'].mean() if 'enrollment' in clinical_data.columns else 0
                        st.metric("Avg Enrollment", f"{avg_enrollment:.0f}")
                        
                except Exception as e:
                    st.error(f"Error fetching clinical trial data: {str(e)}")
        
        if st.session_state.clinical_data is not None:
            st.subheader("Current Clinical Trial Data")
            st.dataframe(st.session_state.clinical_data)
    
    with tab2:
        st.subheader("Stock Data from Yahoo Finance")
        
        # Predefined pharmaceutical companies
        pharma_companies = {
            "Pfizer": "PFE",
            "Johnson & Johnson": "JNJ",
            "Merck": "MRK",
            "AstraZeneca": "AZN",
            "Novartis": "NVS",
            "Roche": "RHHBY",
            "Bristol Myers Squibb": "BMY",
            "AbbVie": "ABBV",
            "Eli Lilly": "LLY",
            "Gilead Sciences": "GILD"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            selected_companies = st.multiselect(
                "Select Pharmaceutical Companies:",
                options=list(pharma_companies.keys()),
                default=["Pfizer", "Johnson & Johnson", "Merck"]
            )
        
        with col2:
            years_back = st.slider("Years of Historical Data:", 1, 10, 5)
        
        if st.button("Fetch Stock Data") and selected_companies:
            with st.spinner("Fetching stock data..."):
                try:
                    tickers = [pharma_companies[company] for company in selected_companies]
                    stock_data = fetch_stock_data(tickers, years_back)
                    st.session_state.stock_data = stock_data
                    
                    st.success(f"Successfully fetched stock data for {len(selected_companies)} companies!")
                    
                    # Display stock data
                    for ticker in tickers:
                        if ticker in stock_data:
                            st.subheader(f"{ticker} Stock Data")
                            st.dataframe(stock_data[ticker].tail())
                            
                            # Stock price chart
                            fig = px.line(
                                stock_data[ticker].reset_index(), 
                                x='Date', 
                                y='Close',
                                title=f"{ticker} Stock Price Over Time"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.write(f"**{ticker} Stock Price Interpretation:**")
                            recent_price = stock_data[ticker]['Close'].iloc[-1]
                            price_change_pct = ((stock_data[ticker]['Close'].iloc[-1] / stock_data[ticker]['Close'].iloc[0]) - 1) * 100
                            st.write(f"""**Simple Explanation:**
                            - **Current Price**: ${recent_price:.2f} per share
                            - **Overall Change**: {'+' if price_change_pct > 0 else ''}{price_change_pct:.1f}% over the time period
                            - **Trend**: The line shows how the company's stock value has changed over time
                            - **Investment Insight**: {'Rising trend suggests investor confidence' if price_change_pct > 0 else 'Declining trend may indicate challenges' if price_change_pct < -10 else 'Relatively stable performance'}
                            """)
                            
                except Exception as e:
                    st.error(f"Error fetching stock data: {str(e)}")
    
    with tab3:
        st.subheader("Simulated Trial Outcomes")
        
        if st.session_state.clinical_data is not None:
            st.info("Generating simulated trial outcomes based on industry statistics...")
            
            if st.button("Generate Simulated Outcomes"):
                with st.spinner("Creating simulated outcomes..."):
                    try:
                        outcomes_data = create_simulated_outcomes(st.session_state.clinical_data)
                        
                        # Merge with clinical data
                        merged_data = st.session_state.clinical_data.copy()
                        merged_data['trial_outcome'] = outcomes_data['trial_outcome']
                        merged_data['success_probability'] = outcomes_data['success_probability']
                        
                        st.session_state.merged_data = merged_data
                        
                        st.success("Simulated outcomes generated successfully!")
                        
                        # Show outcome distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            outcome_counts = merged_data['trial_outcome'].value_counts()
                            fig = px.pie(
                                values=outcome_counts.values,
                                names=['Failure', 'Success'],
                                title="Simulated Trial Outcome Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(
                                merged_data,
                                x='success_probability',
                                title="Distribution of Success Probabilities",
                                nbins=20
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(merged_data.head())
                        
                        # Add interpretation
                        st.write("**Simulated Outcomes Interpretation:**")
                        success_count = outcome_counts.get(1, 0)
                        total_trials = len(merged_data)
                        avg_success_prob = merged_data['success_probability'].mean()
                        
                        st.write(f"""**Key Insights:**
                        - **Success Rate**: {success_count}/{total_trials} trials ({success_count/total_trials:.1%}) were simulated as successful
                        - **Average Success Probability**: {avg_success_prob:.1%} across all trials
                        - **Probability Distribution**: The histogram shows how success probabilities vary across trials
                        - **Industry Reality**: These simulated outcomes reflect real-world pharmaceutical success rates, which are typically low due to the high risk and complexity of drug development
                        
                        The simulation considers factors like trial phase, sponsor type, and enrollment size to create realistic outcome predictions.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error generating simulated outcomes: {str(e)}")
        else:
            st.warning("Please fetch clinical trial data first.")

def show_exploratory_analysis():
    st.header("🔍 Exploratory Data Analysis")
    
    if st.session_state.merged_data is None:
        st.warning("Please complete data acquisition first.")
        return
    
    data = st.session_state.merged_data
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Trial Analysis", "Sponsor Analysis", "Correlations"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
            st.write("**Success Rate:**")
            success_rate = data['trial_outcome'].mean()
            st.metric("Overall Success Rate", f"{success_rate:.1%}")
        
        with col2:
            st.write("**Missing Values:**")
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                st.dataframe(missing_data[missing_data > 0])
            else:
                st.write("No missing values found.")
        
        st.subheader("Data Sample")
        st.dataframe(data.head(10))
    
    with tab2:
        st.subheader("Trial Characteristics Analysis")
        
        if 'phase' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate by phase
                phase_success = data.groupby('phase')['trial_outcome'].agg(['count', 'mean']).reset_index()
                phase_success.columns = ['Phase', 'Trial_Count', 'Success_Rate']
                
                fig = px.bar(
                    phase_success,
                    x='Phase',
                    y='Success_Rate',
                    title='Success Rate by Trial Phase',
                    color='Success_Rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Success Rate by Phase Interpretation:**")
                best_phase = phase_success.loc[phase_success['Success_Rate'].idxmax(), 'Phase']
                worst_phase = phase_success.loc[phase_success['Success_Rate'].idxmin(), 'Phase']
                st.write(f"""**Key Insights:**
                - **{best_phase}** shows the highest success rate ({phase_success['Success_Rate'].max():.1%})
                - **{worst_phase}** shows the lowest success rate ({phase_success['Success_Rate'].min():.1%})
                - Later phases typically have higher success rates as unsuccessful treatments are filtered out
                - Phase 3 trials usually have the highest success rates due to prior validation
                """)
            
            with col2:
                # Trial count by phase
                fig = px.bar(
                    phase_success,
                    x='Phase',
                    y='Trial_Count',
                    title='Number of Trials by Phase'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Trial Count by Phase Interpretation:**")
                most_common_phase = phase_success.loc[phase_success['Trial_Count'].idxmax(), 'Phase']
                st.write(f"""**Simple Explanation:**
                - **{most_common_phase}** has the most trials in our dataset
                - **Early phases** (Phase 1, 2) typically have more trials because many drugs don't make it to later phases
                - **Later phases** (Phase 3, 4) have fewer trials because only promising drugs advance this far
                - **Pattern**: This shows how the pharmaceutical "funnel" works - many drugs start, few finish
                """)
        
        if 'enrollment' in data.columns:
            # Enrollment analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    data,
                    x='enrollment',
                    color='trial_outcome',
                    title='Trial Enrollment Distribution by Outcome',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Enrollment Distribution Interpretation:**")
                avg_enrollment_success = data[data['trial_outcome']==1]['enrollment'].mean()
                avg_enrollment_failure = data[data['trial_outcome']==0]['enrollment'].mean()
                st.write(f"""**Simple Explanation:**
                - **Red bars**: Failed trials, **Blue bars**: Successful trials
                - **Average enrollment for successful trials**: {avg_enrollment_success:.0f} patients
                - **Average enrollment for failed trials**: {avg_enrollment_failure:.0f} patients
                - **Key insight**: {'Successful trials tend to be larger' if avg_enrollment_success > avg_enrollment_failure else 'Trial size varies between successful and failed studies'}
                - **Why this matters**: Larger trials generally provide more reliable results
                """)
            
            with col2:
                # Success rate by enrollment size
                data['enrollment_category'] = pd.cut(
                    data['enrollment'], 
                    bins=[0, 100, 500, 1000, float('inf')], 
                    labels=['Small (<100)', 'Medium (100-500)', 'Large (500-1000)', 'Very Large (>1000)']
                )
                
                enrollment_success = data.groupby('enrollment_category')['trial_outcome'].mean().reset_index()
                
                fig = px.bar(
                    enrollment_success,
                    x='enrollment_category',
                    y='trial_outcome',
                    title='Success Rate by Enrollment Size'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Enrollment Size vs Success Interpretation:**")
                st.write("""**Key Insights:**
                - **Large trials** tend to have higher success rates due to better statistical power
                - **Small trials** may have more variable results due to limited sample sizes
                - **Industry Pattern**: Larger pharmaceutical companies typically run larger trials with better outcomes
                - **Statistical Significance**: Larger enrollments provide more reliable results and regulatory confidence
                """)
    
    with tab3:
        st.subheader("Sponsor Analysis")
        
        if 'sponsor' in data.columns:
            # Top sponsors by trial count
            sponsor_stats = data.groupby('sponsor').agg({
                'trial_outcome': ['count', 'mean'],
                'enrollment': 'mean'
            }).round(3)
            
            sponsor_stats.columns = ['Trial_Count', 'Success_Rate', 'Avg_Enrollment']
            sponsor_stats = sponsor_stats.sort_values('Trial_Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Sponsors by Trial Count")
                top_sponsors = sponsor_stats.head(10).reset_index()
                
                fig = px.bar(
                    top_sponsors,
                    x='Trial_Count',
                    y='sponsor',
                    orientation='h',
                    title='Sponsors by Number of Trials'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Top Sponsors Interpretation:**")
                top_sponsor = top_sponsors.iloc[0]['sponsor'] if len(top_sponsors) > 0 else "N/A"
                st.write(f"""**Key Insights:**
                - **{top_sponsor}** is the most active sponsor with the highest number of trials
                - **Large pharmaceutical companies** typically dominate clinical trial activity
                - **Research volume** often correlates with company size and R&D investment
                - **Market leaders** tend to have diversified pipelines across multiple therapeutic areas
                """)
            
            with col2:
                st.subheader("Success Rate vs Trial Count")
                # Filter sponsors with at least 5 trials for meaningful analysis
                meaningful_sponsors = sponsor_stats[sponsor_stats['Trial_Count'] >= 5].reset_index()
                
                if len(meaningful_sponsors) > 0:
                    fig = px.scatter(
                        meaningful_sponsors,
                        x='Trial_Count',
                        y='Success_Rate',
                        size='Avg_Enrollment',
                        hover_name='sponsor',
                        title='Sponsor Success Rate vs Trial Experience'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("**Sponsor Experience vs Success Interpretation:**")
                    st.write("""**Key Insights:**
                    - **Bubble size** represents average enrollment size (larger = bigger trials)
                    - **Experience matters**: Sponsors with more trials often show improved success rates
                    - **Learning curve**: Organizations improve their trial design and execution over time
                    - **Resource advantage**: Experienced sponsors often have better infrastructure and expertise
                    - **Risk management**: Veteran companies are better at selecting viable drug candidates
                    """)
                else:
                    st.info("Not enough sponsors with sufficient trial data for meaningful analysis.")
    
    with tab4:
        st.subheader("Feature Correlations")
        
        # Select numerical columns for correlation analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            correlation_matrix = data[numerical_cols].corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig)
            
            st.write("**Correlation Matrix Interpretation:**")
            st.write("""This heatmap shows how different features relate to each other:
            - **Red colors** indicate positive correlations (features move together)
            - **Blue colors** indicate negative correlations (features move in opposite directions)  
            - **White/neutral** colors indicate little to no correlation
            - Strong correlations (>0.7 or <-0.7) may indicate redundant features or important relationships
            """)
            
            # Feature importance for trial outcome
            if 'trial_outcome' in numerical_cols:
                outcome_corr = correlation_matrix['trial_outcome'].abs().sort_values(ascending=False)
                outcome_corr = outcome_corr.drop('trial_outcome')  # Remove self-correlation
                
                if len(outcome_corr) > 0:
                    st.subheader("Features Most Correlated with Trial Outcome")
                    
                    fig = px.bar(
                        x=outcome_corr.values,
                        y=outcome_corr.index,
                        orientation='h',
                        title='Absolute Correlation with Trial Outcome'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical features for correlation analysis.")

def show_predictive_modeling():
    st.header("🤖 Predictive Modeling")
    
    if st.session_state.merged_data is None:
        st.warning("Please complete data acquisition first.")
        return
    
    data = st.session_state.merged_data
    
    tab1, tab2, tab3 = st.tabs(["Feature Engineering", "Model Training", "Model Evaluation"])
    
    with tab1:
        st.subheader("Feature Engineering")
        
        # Prepare features
        try:
            X, y, feature_names = prepare_features(data)
            
            if X is not None:
                st.success(f"Successfully prepared {X.shape[1]} features from {X.shape[0]} samples")
                
                # Feature summary
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Type': ['Numerical' if X[:, i].dtype in [np.float64, np.int64] else 'Categorical' 
                            for i in range(len(feature_names))]
                })
                
                st.subheader("Feature Summary")
                st.dataframe(feature_df)
                
                # Target distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    target_dist = pd.Series(y).value_counts()
                    fig = px.pie(
                        values=target_dist.values,
                        names=['Failure', 'Success'],
                        title="Target Variable Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Class Balance", f"{target_dist[1]}/{target_dist[0]}")
                    success_rate = np.mean(y) if len(y) > 0 and not np.isnan(np.mean(y)) else 0
                    st.metric("Success Rate", f"{success_rate:.1%}")
                    
                    st.write("**Target Variable Interpretation:**")
                    st.write(f"""This chart shows the distribution of trial outcomes in our dataset. 
                    Out of {len(y)} trials, {target_dist.get(1, 0)} were successful and {target_dist.get(0, 0)} failed. 
                    The success rate of {success_rate:.1%} reflects the realistic challenges in pharmaceutical 
                    development, where many trials don't meet their primary endpoints.""")
            else:
                st.error("Failed to prepare features. Check your data.")
                
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
    
    with tab2:
        st.subheader("Model Training")
        
        if st.button("Train Predictive Model"):
            try:
                X, y, feature_names = prepare_features(data)
                
                if X is not None and len(y) > 0 and not np.any(np.isnan(y)):
                    with st.spinner("Training model..."):
                        model, metrics = train_model(X, y, feature_names)
                        
                        st.session_state.model = model
                        st.session_state.model_metrics = metrics
                        st.session_state.feature_names = feature_names
                        
                        st.success("Model trained successfully!")
                        
                        # Display basic metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                            
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
        
        if st.session_state.model is not None:
            st.success("Model is ready for evaluation and predictions!")
    
    with tab3:
        st.subheader("Model Evaluation")
        
        if st.session_state.model is None:
            st.warning("Please train a model first.")
            return
        
        metrics = st.session_state.model_metrics
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = metrics['confusion_matrix']
            
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
            st.write("**Confusion Matrix Interpretation:**")
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            accuracy = (tp + tn) / total
            st.write(f"""**Simple Explanation:**
            - **Top-left ({tn})**: Correctly predicted failures (Good!)
            - **Top-right ({fp})**: Wrongly predicted as failures (Missed opportunities)
            - **Bottom-left ({fn})**: Wrongly predicted as successes (False hopes)
            - **Bottom-right ({tp})**: Correctly predicted successes (Great!)
            - **Overall Accuracy**: {accuracy:.1%} of predictions were correct
            - **What this means**: The darker the blue, the more predictions in that category
            """)
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
            
            fig = px.line(
                x=fpr, 
                y=tpr, 
                title=f'ROC Curve (AUC = {metrics["auc"]:.3f})'
            )
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="red", dash="dash")
            )
            fig.update_xaxes(title="False Positive Rate")
            fig.update_yaxes(title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**ROC Curve Interpretation:**")
            auc_score = metrics['auc']
            if auc_score > 0.9:
                performance = "Excellent"
            elif auc_score > 0.8:
                performance = "Good"
            elif auc_score > 0.7:
                performance = "Fair"
            else:
                performance = "Poor"
                
            st.write(f"""The ROC curve measures how well our model distinguishes between successful and failed trials:
            - **AUC Score**: {auc_score:.3f} (closer to 1.0 is better)
            - **Performance**: {performance} discrimination ability
            - **Red dashed line**: Random chance (AUC = 0.5)
            - **Interpretation**: The model can {'very effectively' if auc_score > 0.8 else 'moderately' if auc_score > 0.7 else 'somewhat'} distinguish between trials that will succeed vs. fail
            """)
        
        # Feature Importance
        if 'feature_importance' in metrics:
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': metrics['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Feature Importance Interpretation:**")
            top_feature = importance_df.iloc[0]['Feature'] if len(importance_df) > 0 else "N/A"
            st.write(f"""**Simple Explanation:**
            - **{top_feature}** is the most important factor for predicting trial success
            - **Longer bars** = More important for making predictions
            - **What this tells us**: These are the key factors that determine if a trial will succeed
            - **For investors**: Focus on trials with favorable characteristics in these top features
            - **Real-world impact**: Companies should pay attention to these factors when designing trials
            """)
            
            # Detailed feature importance table
            st.subheader("Detailed Feature Importance")
            st.dataframe(importance_df)

def show_investment_insights():
    st.header("💰 Investment Insights")
    
    if st.session_state.model is None:
        st.warning("Please train a predictive model first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Portfolio Optimization", "Scenario Analysis"])
    
    with tab1:
        st.subheader("Risk Analysis")
        
        data = st.session_state.merged_data
        
        if 'sponsor' in data.columns:
            # Sponsor risk analysis
            sponsor_risk = data.groupby('sponsor').agg({
                'trial_outcome': ['count', 'mean', 'std'],
                'success_probability': 'mean'
            }).round(3)
            
            sponsor_risk.columns = ['Trial_Count', 'Success_Rate', 'Volatility', 'Avg_Success_Prob']
            sponsor_risk['Risk_Score'] = (1 - sponsor_risk['Success_Rate']) * sponsor_risk['Volatility']
            sponsor_risk = sponsor_risk.fillna(0)
            
            # Filter sponsors with meaningful data
            sponsor_risk = sponsor_risk[sponsor_risk['Trial_Count'] >= 3]
            
            if len(sponsor_risk) > 0:
                st.subheader("Sponsor Risk-Return Analysis")
                
                fig = px.scatter(
                    sponsor_risk.reset_index(),
                    x='Risk_Score',
                    y='Success_Rate',
                    size='Trial_Count',
                    hover_name='sponsor',
                    title='Risk vs Success Rate by Sponsor',
                    labels={'Risk_Score': 'Risk Score (Lower is Better)', 
                           'Success_Rate': 'Historical Success Rate'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Risk-Return Analysis Interpretation:**")
                st.write("""**Simple Explanation:**
                - **Best investments**: Top-left area (low risk, high success rate)
                - **Avoid**: Bottom-right area (high risk, low success rate)
                - **Bubble size**: Number of trials (bigger = more experience)
                - **Smart investing**: Look for sponsors with consistently low risk and high success rates
                - **What this means**: Choose partners with proven track records and stable performance
                """)
                
                # Top and bottom performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Lowest Risk Sponsors")
                    low_risk = sponsor_risk.sort_values('Risk_Score').head(5)
                    st.dataframe(low_risk)
                
                with col2:
                    st.subheader("⚠️ Highest Risk Sponsors")
                    high_risk = sponsor_risk.sort_values('Risk_Score', ascending=False).head(5)
                    st.dataframe(high_risk)
        
        # Cost analysis
        st.subheader("Development Cost Analysis")
        
        # Simulated cost data based on industry averages
        phase_costs = {
            'Phase 1': 15.6,  # Million USD
            'Phase 2': 46.4,
            'Phase 3': 255.4,
            'Phase 4': 20.0
        }
        
        if 'phase' in data.columns:
            cost_analysis = data.groupby('phase').agg({
                'trial_outcome': ['count', 'mean']
            }).round(3)
            
            cost_analysis.columns = ['Trial_Count', 'Success_Rate']
            cost_analysis['Avg_Cost_Million'] = [phase_costs.get(phase, 50) for phase in cost_analysis.index]
            cost_analysis['Expected_ROI'] = cost_analysis['Success_Rate'] * 1000 - cost_analysis['Avg_Cost_Million']  # Assuming $1B revenue for successful drug
            
            fig = px.bar(
                cost_analysis.reset_index(),
                x='phase',
                y='Expected_ROI',
                title='Expected ROI by Trial Phase (Million USD)',
                color='Expected_ROI',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Expected ROI Interpretation:**")
            best_roi_phase = cost_analysis.loc[cost_analysis['Expected_ROI'].idxmax()].name if len(cost_analysis) > 0 else "N/A"
            st.write(f"""**Simple Explanation:**
            - **Green bars**: Phases with positive expected returns (profitable)
            - **Red bars**: Phases with negative expected returns (likely losses)
            - **{best_roi_phase}** offers the best expected return on investment
            - **Higher bars**: Better investment opportunity
            - **Investment strategy**: Focus on phases with consistently positive ROI
            - **Reality check**: These are estimates based on average industry success rates and costs
            """)
    
    with tab2:
        st.subheader("Portfolio Optimization")
        
        st.info("Portfolio optimization based on predicted success probabilities")
        
        # Investment parameters
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Total Investment Amount (Million USD):", 
                min_value=1, max_value=1000, value=100
            )
            
        with col2:
            max_risk_tolerance = st.slider(
                "Risk Tolerance (0=Conservative, 1=Aggressive):", 
                0.0, 1.0, 0.5
            )
        
        if st.button("Optimize Portfolio"):
            data = st.session_state.merged_data
            
            # Create investment opportunities
            opportunities = []
            
            for idx, row in data.iterrows():
                if row['success_probability'] > max_risk_tolerance:
                    opportunities.append({
                        'Trial_ID': idx,
                        'Sponsor': row.get('sponsor', 'Unknown'),
                        'Phase': row.get('phase', 'Unknown'),
                        'Success_Probability': row['success_probability'],
                        'Estimated_Cost': phase_costs.get(row.get('phase', 'Unknown'), 50),
                        'Expected_Value': row['success_probability'] * 1000  # Expected revenue
                    })
            
            if opportunities:
                opportunities_df = pd.DataFrame(opportunities)
                opportunities_df['ROI_Ratio'] = opportunities_df['Expected_Value'] / opportunities_df['Estimated_Cost']
                opportunities_df = opportunities_df.sort_values('ROI_Ratio', ascending=False)
                
                st.subheader("🎯 Recommended Investment Opportunities")
                st.dataframe(opportunities_df.head(10))
                
                # Portfolio allocation
                total_cost = opportunities_df.head(10)['Estimated_Cost'].sum()
                if total_cost <= investment_amount:
                    allocation = opportunities_df.head(10).copy()
                    allocation['Allocation_Percent'] = (allocation['Estimated_Cost'] / allocation['Estimated_Cost'].sum()) * 100
                    
                    fig = px.pie(
                        allocation,
                        values='Allocation_Percent',
                        names='Sponsor',
                        title=f'Recommended Portfolio Allocation ({investment_amount}M USD)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Portfolio metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_success_prob = allocation['Success_Probability'].mean()
                        st.metric("Portfolio Success Probability", f"{avg_success_prob:.1%}")
                    with col2:
                        expected_return = (allocation['Expected_Value'] * allocation['Success_Probability']).sum()
                        st.metric("Expected Return (Million USD)", f"${expected_return:.1f}M")
                    with col3:
                        portfolio_roi = (expected_return - investment_amount) / investment_amount
                        st.metric("Expected Portfolio ROI", f"{portfolio_roi:.1%}")
                else:
                    st.warning(f"Recommended opportunities require ${total_cost:.1f}M, but budget is ${investment_amount}M")
            else:
                st.info("No suitable investment opportunities found with current risk tolerance.")
    
    with tab3:
        st.subheader("Scenario Analysis")
        
        st.info("Analyze different market scenarios and their impact on investment returns")
        
        # Scenario parameters
        col1, col2 = st.columns(2)
        
        with col1:
            market_condition = st.selectbox(
                "Market Condition:",
                ["Bull Market", "Bear Market", "Stable Market", "Recession"]
            )
            
        with col2:
            regulatory_environment = st.selectbox(
                "Regulatory Environment:",
                ["Favorable", "Standard", "Strict", "Very Strict"]
            )
        
        # Scenario multipliers
        market_multipliers = {
            "Bull Market": 1.2,
            "Bear Market": 0.8,
            "Stable Market": 1.0,
            "Recession": 0.6
        }
        
        regulatory_multipliers = {
            "Favorable": 1.15,
            "Standard": 1.0,
            "Strict": 0.9,
            "Very Strict": 0.75
        }
        
        if st.button("Run Scenario Analysis"):
            market_mult = market_multipliers[market_condition]
            regulatory_mult = regulatory_multipliers[regulatory_environment]
            combined_mult = market_mult * regulatory_mult
            
            # Base case vs scenario
            base_success_rate = st.session_state.merged_data['trial_outcome'].mean()
            scenario_success_rate = min(base_success_rate * combined_mult, 1.0)
            
            scenario_results = pd.DataFrame({
                'Metric': [
                    'Success Rate',
                    'Expected ROI',
                    'Risk Level',
                    'Market Attractiveness'
                ],
                'Base Case': [
                    f"{base_success_rate:.1%}",
                    "12.5%",
                    "Medium",
                    "Moderate"
                ],
                'Scenario': [
                    f"{scenario_success_rate:.1%}",
                    f"{12.5 * combined_mult:.1f}%",
                    "High" if combined_mult < 0.8 else "Low" if combined_mult > 1.1 else "Medium",
                    "High" if combined_mult > 1.1 else "Low" if combined_mult < 0.8 else "Moderate"
                ]
            })
            
            st.subheader("Scenario Analysis Results")
            st.dataframe(scenario_results)
            
            # Visual comparison
            comparison_data = pd.DataFrame({
                'Scenario': ['Base Case', f'{market_condition} + {regulatory_environment}'],
                'Success_Rate': [base_success_rate, scenario_success_rate],
                'Expected_ROI': [0.125, 0.125 * combined_mult]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comparison_data,
                    x='Scenario',
                    y='Success_Rate',
                    title='Success Rate Comparison'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    comparison_data,
                    x='Scenario',
                    y='Expected_ROI',
                    title='Expected ROI Comparison'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Investment Recommendations")
            
            if combined_mult > 1.1:
                st.success("🚀 **Strong Buy Signal**: Market conditions are very favorable for pharmaceutical investments.")
            elif combined_mult > 0.9:
                st.info("📈 **Buy Signal**: Market conditions are favorable for selective investments.")
            elif combined_mult > 0.8:
                st.warning("⚠️ **Hold Signal**: Market conditions are challenging, consider defensive positions.")
            else:
                st.error("🛑 **Sell Signal**: Market conditions are very unfavorable, consider reducing exposure.")

if __name__ == "__main__":
    main()
