import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Experiment Incremental Impact Report")

# Password protection
if "authenticated" not in st.session_state:
    password = st.text_input("Enter Password:", type="password")
    if password == "Antidormancy@68C":
        st.session_state["authenticated"] = True
    else:
        st.stop()

st.title("📈 Experiment Incremental Impact Report")
st.markdown("### Key performance metrics and incremental value")

# Backend File Upload (Assumed to be fixed)
uploaded_file = "1MG_Test_and_control_report_transformed (2).csv"  # Replace with actual backend file path

@st.cache_data
def load_data():
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    
    # Sort data by date
    df = df.sort_values(by='date')
    
    # Calculate metrics
    df['gmv_per_audience'] = df['gmv'] / df['audience_size']
    df['app_opens_per_audience'] = df['app_opens'] / df['audience_size']
    df['orders_per_audience'] = df['orders'] / df['audience_size']
    df['transactors_per_audience'] = df['transactors'] / df['audience_size']
    
    return df

df = load_data()

# Check for Recency column
has_recency_data = 'Recency' in df.columns

# Define control group
control_group = "Control Set"

# Test start dates
test_start_dates = {
    "resp": pd.Timestamp("2025-03-05"),
    "cardiac": pd.Timestamp("2025-03-18"),
    "diabetic": pd.Timestamp("2025-03-06"),
    "derma": pd.Timestamp("2025-03-18")
}

# Define recency ranges
recency_ranges = ['91-120', '121-150', '151-180', '181-365']

# Cohort descriptions
cohort_descriptions = {
    "resp": "Respiratory Health",
    "cardiac": "Cardiac Health",
    "diabetic": "Diabetes Management",
    "derma": "Dermatology Care"
}

# Sidebar for filtering
st.sidebar.header("Filters")
view_option = st.sidebar.radio("View Level", ["Executive Summary", "Detailed Analysis"])

if view_option == "Detailed Analysis":
    selected_cohort = st.sidebar.selectbox("Select Cohort", 
                                          df['cohort'].unique(),
                                          format_func=lambda x: cohort_descriptions.get(x, x))
    
    if has_recency_data:
        recency_options = ["All Recencies"] + recency_ranges
        selected_recency = st.sidebar.selectbox("Select Recency", recency_options)
    else:
        selected_recency = "All Recencies"
else:
    selected_cohort = None
    selected_recency = None

# Calculate incremental impact for all cohorts
def calculate_incremental_impact(df, cohort=None, recency=None):
    results = []
    
    cohorts_to_process = [cohort] if cohort else df['cohort'].unique()
    
    for curr_cohort in cohorts_to_process:
        # Get start date
        start_date = test_start_dates.get(curr_cohort)
        if not start_date:
            continue
            
        # Get all test groups for this cohort
        cohort_df = df[df['cohort'] == curr_cohort]
        test_groups = [g for g in cohort_df['data_set'].unique() if g != control_group]
        
        for test_group in test_groups:
            # Filter by recency if specified
            if recency and recency != "All Recencies" and has_recency_data:
                cohort_data = cohort_df[cohort_df['Recency'] == recency]
            else:
                cohort_data = cohort_df
                
            # Get test and control data
            test_data = cohort_data[(cohort_data['data_set'] == test_group) & 
                                  (cohort_data['date'] >= start_date)]
            control_data = cohort_data[(cohort_data['data_set'] == control_group) & 
                                     (cohort_data['date'] >= start_date)]
            
            if len(test_data) == 0 or len(control_data) == 0:
                continue
                
            # Calculate absolute values
            test_audience = test_data['audience_size'].sum()
            control_audience = control_data['audience_size'].sum()
            
            test_gmv = test_data['gmv'].sum()
            control_gmv = control_data['gmv'].sum()
            
            test_app_opens = test_data['app_opens'].sum()
            control_app_opens = control_data['app_opens'].sum()
            
            test_orders = test_data['orders'].sum()
            control_orders = control_data['orders'].sum()
            
            test_transactors = test_data['transactors'].sum()
            control_transactors = control_data['transactors'].sum()
            
            # Calculate scaled values (as per the requested formula)
            # (((total test group for test duration)100/70)-((total control group for test duration)100/30))
            test_gmv_scaled = (test_gmv * 100) / 70
            control_gmv_scaled = (control_gmv * 100) / 30
            incremental_gmv = test_gmv_scaled - control_gmv_scaled
            
            test_app_opens_scaled = (test_app_opens * 100) / 70
            control_app_opens_scaled = (control_app_opens * 100) / 30
            incremental_app_opens = test_app_opens_scaled - control_app_opens_scaled
            
            test_orders_scaled = (test_orders * 100) / 70
            control_orders_scaled = (control_orders * 100) / 30
            incremental_orders = test_orders_scaled - control_orders_scaled
            
            test_transactors_scaled = (test_transactors * 100) / 70
            control_transactors_scaled = (control_transactors * 100) / 30
            incremental_transactors = test_transactors_scaled - control_transactors_scaled
            
            # Calculate per audience metrics
            test_gmv_per_audience = test_gmv / test_audience
            control_gmv_per_audience = control_gmv / control_audience
            gmv_lift_percent = ((test_gmv_per_audience - control_gmv_per_audience) / control_gmv_per_audience) * 100
            
            test_app_opens_per_audience = test_app_opens / test_audience
            control_app_opens_per_audience = control_app_opens / control_audience
            app_opens_lift_percent = ((test_app_opens_per_audience - control_app_opens_per_audience) / control_app_opens_per_audience) * 100
            
            test_orders_per_audience = test_orders / test_audience
            control_orders_per_audience = control_orders / control_audience
            orders_lift_percent = ((test_orders_per_audience - control_orders_per_audience) / control_orders_per_audience) * 100
            
            test_transactors_per_audience = test_transactors / test_audience
            control_transactors_per_audience = control_transactors / control_audience
            transactors_lift_percent = ((test_transactors_per_audience - control_transactors_per_audience) / control_transactors_per_audience) * 100
            
            recency_label = recency if recency and recency != "All Recencies" else "All"
            
            # Collect results
            results.append({
                'Cohort': curr_cohort,
                'Cohort Label': cohort_descriptions.get(curr_cohort, curr_cohort),
                'Test Group': test_group,
                'Recency': recency_label,
                'Test Audience': test_audience,
                'Control Audience': control_audience,
                'Test GMV': test_gmv,
                'Control GMV': control_gmv,
                'Incremental GMV': incremental_gmv,
                'GMV Lift %': gmv_lift_percent,
                'Test App Opens': test_app_opens,
                'Control App Opens': control_app_opens,
                'Incremental App Opens': incremental_app_opens,
                'App Opens Lift %': app_opens_lift_percent,
                'Test Orders': test_orders,
                'Control Orders': control_orders,
                'Incremental Orders': incremental_orders,
                'Orders Lift %': orders_lift_percent,
                'Test Transactors': test_transactors,
                'Control Transactors': control_transactors,
                'Incremental Transactors': incremental_transactors,
                'Transactors Lift %': transactors_lift_percent
            })
    
    return pd.DataFrame(results)

# Get impact results
impact_results = calculate_incremental_impact(df, selected_cohort, selected_recency)

# Display executive summary if selected
if view_option == "Executive Summary":
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overall Impact", "Cohort Analysis", "Recency Analysis", "ROI Projection"])
    
    with tab1:
        st.header("Overall Experiment Impact")
        
        # Calculate total incremental metrics across all cohorts
        total_incremental_gmv = impact_results['Incremental GMV'].sum()
        total_incremental_orders = impact_results['Incremental Orders'].sum()
        total_incremental_app_opens = impact_results['Incremental App Opens'].sum()
        total_incremental_transactors = impact_results['Incremental Transactors'].sum()
        
        # Create KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Incremental GMV", f"₹{total_incremental_gmv:,.2f}")
        
        with col2:
            st.metric("Incremental Orders", f"{total_incremental_orders:,.0f}")
            
        with col3:
            st.metric("Incremental App Opens", f"{total_incremental_app_opens:,.0f}")
            
        with col4:
            st.metric("Incremental Transactors", f"{total_incremental_transactors:,.0f}")
            
        # Average lift percentages
        avg_gmv_lift = impact_results['GMV Lift %'].mean()
        avg_orders_lift = impact_results['Orders Lift %'].mean()
        avg_app_opens_lift = impact_results['App Opens Lift %'].mean()
        avg_transactors_lift = impact_results['Transactors Lift %'].mean()
        
        st.subheader("Average Lift Percentages")
        
        # Create a bar chart for lift percentages
        lift_data = {
            'Metric': ['GMV', 'Orders', 'App Opens', 'Transactors'],
            'Lift %': [avg_gmv_lift, avg_orders_lift, avg_app_opens_lift, avg_transactors_lift]
        }
        
        lift_df = pd.DataFrame(lift_data)
        
        fig = px.bar(lift_df, x='Metric', y='Lift %', text_auto='.2f',
                    color='Lift %', color_continuous_scale='Blugrn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.header("Impact by Medical Concern")
        
        # Group by cohort
        cohort_impact = impact_results.groupby('Cohort Label').agg({
            'Incremental GMV': 'sum',
            'Incremental Orders': 'sum',
            'Incremental App Opens': 'sum',
            'Incremental Transactors': 'sum',
            'GMV Lift %': 'mean',
            'Orders Lift %': 'mean',
            'App Opens Lift %': 'mean',
            'Transactors Lift %': 'mean'
        }).reset_index()
        
        # Create a subplot with 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Incremental GMV by Concern", "Incremental Orders by Concern",
                           "Incremental App Opens by Concern", "Incremental Transactors by Concern")
        )
        
        # Add bars for each metric
        fig.add_trace(
            go.Bar(x=cohort_impact['Cohort Label'], y=cohort_impact['Incremental GMV'],
                  text=cohort_impact['Incremental GMV'].apply(lambda x: f'₹{x:,.2f}'),
                  marker_color='skyblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=cohort_impact['Cohort Label'], y=cohort_impact['Incremental Orders'],
                  text=cohort_impact['Incremental Orders'].apply(lambda x: f'{x:,.0f}'),
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=cohort_impact['Cohort Label'], y=cohort_impact['Incremental App Opens'],
                  text=cohort_impact['Incremental App Opens'].apply(lambda x: f'{x:,.0f}'),
                  marker_color='orange'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=cohort_impact['Cohort Label'], y=cohort_impact['Incremental Transactors'],
                  text=cohort_impact['Incremental Transactors'].apply(lambda x: f'{x:,.0f}'),
                  marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show lift percentages by cohort
        st.subheader("Lift Percentages by Medical Concern")
        
        lift_metrics = ['GMV Lift %', 'Orders Lift %', 'App Opens Lift %', 'Transactors Lift %']
        lift_by_cohort = pd.melt(cohort_impact, 
                               id_vars=['Cohort Label'], 
                               value_vars=lift_metrics,
                               var_name='Metric', 
                               value_name='Lift %')
        
        fig = px.bar(lift_by_cohort, x='Cohort Label', y='Lift %', color='Metric', barmode='group',
                    text_auto='.2f')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.header("Impact by Customer Recency")
        
        if has_recency_data:
            # Filter for records with specific recency values
            recency_impact = impact_results[impact_results['Recency'].isin(recency_ranges)]
            
            if len(recency_impact) > 0:
                # Group by recency
                recency_grouped = recency_impact.groupby('Recency').agg({
                    'Incremental GMV': 'sum',
                    'Incremental Orders': 'sum',
                    'Incremental App Opens': 'sum',
                    'Incremental Transactors': 'sum',
                    'GMV Lift %': 'mean',
                    'Orders Lift %': 'mean',
                    'App Opens Lift %': 'mean',
                    'Transactors Lift %': 'mean'
                }).reset_index()
                
                # Sort by recency
                recency_grouped['Recency'] = pd.Categorical(recency_grouped['Recency'], 
                                                          categories=recency_ranges, 
                                                          ordered=True)
                recency_grouped = recency_grouped.sort_values('Recency')
                
                # Visualize incremental GMV by recency
                st.subheader("Incremental GMV by Customer Recency")
                fig = px.bar(recency_grouped, x='Recency', y='Incremental GMV',
                            text_auto=lambda v: f'₹{v:,.2f}',
                            color='Incremental GMV',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualize lift percentages by recency
                st.subheader("Lift Percentages by Customer Recency")
                
                lift_metrics = ['GMV Lift %', 'Orders Lift %', 'App Opens Lift %', 'Transactors Lift %']
                lift_by_recency = pd.melt(recency_grouped, 
                                       id_vars=['Recency'], 
                                       value_vars=lift_metrics,
                                       var_name='Metric', 
                                       value_name='Lift %')
                
                fig = px.line(lift_by_recency, x='Recency', y='Lift %', color='Metric', markers=True,
                             line_shape='spline')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recency data available for analysis")
        else:
            st.info("Recency data not available in the dataset")
    
    with tab4:
        st.header("ROI Projection")
        
        # Calculate approximate ROI metrics
        total_test_audience = impact_results['Test Audience'].sum()
        total_control_audience = impact_results['Control Audience'].sum()
        
        # Assume some costs for the campaign
        st.info("Assuming campaign costs for ROI calculation.")
        
        campaign_cost = st.number_input("Enter campaign cost (₹)", 
                                       min_value=10000, 
                                       max_value=10000000, 
                                       value=500000,
                                       step=10000)
        
        # Calculate ROI
        roi_percentage = (total_incremental_gmv - campaign_cost) / campaign_cost * 100
        
        # Display ROI metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Campaign Cost", f"₹{campaign_cost:,.2f}")
        
        with col2:
            st.metric("Incremental Revenue", f"₹{total_incremental_gmv:,.2f}")
            
        with col3:
            st.metric("ROI", f"{roi_percentage:.2f}%")
        
        # Create a gauge chart for ROI
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = roi_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Return on Investment"},
            gauge = {
                'axis': {'range': [None, max(300, roi_percentage * 1.2)]},
                'steps': [
                    {'range': [0, 100], 'color': "lightgray"},
                    {'range': [100, 200], 'color': "lightgreen"},
                    {'range': [200, 300], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional ROI insights
        st.subheader("ROI by Medical Concern")
        
        # Calculate ROI for each cohort
        cohort_roi = cohort_impact.copy()
        cohort_roi['Allocated Cost'] = campaign_cost / len(cohort_roi)
        cohort_roi['ROI %'] = (cohort_roi['Incremental GMV'] - cohort_roi['Allocated Cost']) / cohort_roi['Allocated Cost'] * 100
        
        fig = px.bar(cohort_roi, x='Cohort Label', y='ROI %',
                    text_auto='.2f',
                    color='ROI %',
                    color_continuous_scale='RdYlGn')
                    
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cohort ROI table
        cohort_roi_display = cohort_roi[['Cohort Label', 'Incremental GMV', 'Allocated Cost', 'ROI %']]
        cohort_roi_display = cohort_roi_display.round(2)
        st.dataframe(cohort_roi_display, use_container_width=True)

# Display detailed analysis if selected
else:
    if selected_cohort:
        st.header(f"Detailed Analysis: {cohort_descriptions.get(selected_cohort, selected_cohort)}")
        
        if selected_recency and selected_recency != "All Recencies":
            st.subheader(f"Recency Segment: {selected_recency} days")
        
        # Filter the results for the selected cohort and recency
        filtered_results = impact_results[impact_results['Cohort'] == selected_cohort]
        
        if selected_recency and selected_recency != "All Recencies":
            filtered_results = filtered_results[filtered_results['Recency'] == selected_recency]
        
        if len(filtered_results) > 0:
            # Display incremental metrics
            st.subheader("Incremental Impact")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Incremental GMV", 
                         f"₹{filtered_results['Incremental GMV'].sum():,.2f}", 
                         f"{filtered_results['GMV Lift %'].mean():.2f}%")
            
            with col2:
                st.metric("Incremental Orders", 
                         f"{filtered_results['Incremental Orders'].sum():,.0f}", 
                         f"{filtered_results['Orders Lift %'].mean():.2f}%")
                
            with col3:
                st.metric("Incremental App Opens", 
                         f"{filtered_results['Incremental App Opens'].sum():,.0f}", 
                         f"{filtered_results['App Opens Lift %'].mean():.2f}%")
                
            with col4:
                st.metric("Incremental Transactors", 
                         f"{filtered_results['Incremental Transactors'].sum():,.0f}", 
                         f"{filtered_results['Transactors Lift %'].mean():.2f}%")
            
            # Show raw data comparison
            st.subheader("Test vs Control Comparison")
            
            # Create comparison tables
            comparison_data = []
            for index, row in filtered_results.iterrows():
                comparison_data.append({
                    'Metric': 'GMV',
                    'Test': f"₹{row['Test GMV']:,.2f}",
                    'Control': f"₹{row['Control GMV']:,.2f}",
                    'Test (Scaled to 100%)': f"₹{row['Test GMV'] * 100/70:,.2f}",
                    'Control (Scaled to 100%)': f"₹{row['Control GMV'] * 100/30:,.2f}",
                    'Incremental': f"₹{row['Incremental GMV']:,.2f}",
                    'Lift %': f"{row['GMV Lift %']:.2f}%"
                })
                
                comparison_data.append({
                    'Metric': 'App Opens',
                    'Test': f"{row['Test App Opens']:,.0f}",
                    'Control': f"{row['Control App Opens']:,.0f}",
                    'Test (Scaled to 100%)': f"{row['Test App Opens'] * 100/70:,.0f}",
                    'Control (Scaled to 100%)': f"{row['Control App Opens'] * 100/30:,.0f}",
                    'Incremental': f"{row['Incremental App Opens']:,.0f}",
                    'Lift %': f"{row['App Opens Lift %']:.2f}%"
                })
                
                comparison_data.append({
                    'Metric': 'Orders',
                    'Test': f"{row['Test Orders']:,.0f}",
                    'Control': f"{row['Control Orders']:,.0f}",
                    'Test (Scaled to 100%)': f"{row['Test Orders'] * 100/70:,.0f}",
                    'Control (Scaled to 100%)': f"{row['Control Orders'] * 100/30:,.0f}",
                    'Incremental': f"{row['Incremental Orders']:,.0f}",
                    'Lift %': f"{row['Orders Lift %']:.2f}%"
                })
                
                comparison_data.append({
                    'Metric': 'Transactors',
                    'Test': f"{row['Test Transactors']:,.0f}",
                    'Control': f"{row['Control Transactors']:,.0f}",
                    'Test (Scaled to 100%)': f"{row['Test Transactors'] * 100/70:,.0f}",
                    'Control (Scaled to 100%)': f"{row['Control Transactors'] * 100/30:,.0f}",
                    'Incremental': f"{row['Incremental Transactors']:,.0f}",
                    'Lift %': f"{row['Transactors Lift %']:.2f}%"
                })
                
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Audience details
            st.subheader("Audience Details")
            
            audience_data = filtered_results[['Test Group', 'Test Audience', 'Control Audience']].reset_index(drop=True)
            
            # Add 70/30 split information
            audience_data['Test Split'] = "70% of total audience"
            audience_data['Control Split'] = "30% of total audience"
            
            st.dataframe(audience_data, use_container_width=True)
            
            # If recency data is available, show breakdown by recency
            if has_recency_data and selected_recency == "All Recencies":
                st.subheader("Performance by Recency Segment")
                
                # Get recency breakdown for this cohort
                recency_breakdown = impact_results[(impact_results['Cohort'] == selected_cohort) & 
                                                 (impact_results['Recency'].isin(recency_ranges))]
                
                if len(recency_breakdown) > 0:
                    # Sort by recency
                    recency_breakdown['Recency'] = pd.Categorical(recency_breakdown['Recency'], 
                                                                categories=recency_ranges, 
                                                                ordered=True)
                    recency_breakdown = recency_breakdown.sort_values('Recency')
                    
# Show GMV impact by recency
                    fig = px.bar(recency_breakdown, x='Recency', y='Incremental GMV',
                                text_auto=lambda v: f'₹{v:,.2f}',
                                color='Incremental GMV',
                                color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show lift percentage by recency
                    fig = px.line(recency_breakdown, x='Recency', y=['GMV Lift %', 'Orders Lift %', 'App Opens Lift %', 'Transactors Lift %'],
                                markers=True, line_shape='spline')
                    fig.update_layout(height=400, title="Lift Percentages by Recency", yaxis_title="Lift %")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed recency data
                    recency_details = recency_breakdown[['Recency', 'Incremental GMV', 'GMV Lift %', 
                                                      'Incremental Orders', 'Orders Lift %']]
                    st.dataframe(recency_details.round(2), use_container_width=True)
                else:
                    st.info("No recency breakdown data available for this cohort")
            
            # Show daily trend analysis
            st.subheader("Daily Performance Trend")
            
            # Get the cohort's test start date
            cohort_start_date = test_start_dates.get(selected_cohort)
            
            if cohort_start_date:
                # Filter data for selected cohort and time period
                daily_data = df[(df['cohort'] == selected_cohort) & (df['date'] >= cohort_start_date)]
                
                # Create daily trend for GMV per audience
                daily_trend = daily_data.pivot_table(
                    index='date',
                    columns='data_set',
                    values='gmv_per_audience',
                    aggfunc='sum'
                ).reset_index()
                
                # Get test and control columns
                test_columns = [col for col in daily_trend.columns if col != 'date' and col != control_group]
                
                if len(test_columns) > 0 and control_group in daily_trend.columns:
                    # Create line chart for GMV per audience
                    fig = go.Figure()
                    
                    # Add control line
                    fig.add_trace(go.Scatter(
                        x=daily_trend['date'],
                        y=daily_trend[control_group],
                        mode='lines+markers',
                        name=control_group,
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add test group lines
                    for test_col in test_columns:
                        fig.add_trace(go.Scatter(
                            x=daily_trend['date'],
                            y=daily_trend[test_col],
                            mode='lines+markers',
                            name=test_col
                        ))
                    
                    fig.update_layout(
                        title='Daily GMV per Audience Member',
                        xaxis_title='Date',
                        yaxis_title='GMV per Audience Member (₹)',
                        legend_title='Group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show orders per audience trend
                    orders_trend = daily_data.pivot_table(
                        index='date',
                        columns='data_set',
                        values='orders_per_audience',
                        aggfunc='sum'
                    ).reset_index()
                    
                    fig = go.Figure()
                    
                    # Add control line
                    fig.add_trace(go.Scatter(
                        x=orders_trend['date'],
                        y=orders_trend[control_group],
                        mode='lines+markers',
                        name=control_group,
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add test group lines
                    for test_col in test_columns:
                        fig.add_trace(go.Scatter(
                            x=orders_trend['date'],
                            y=orders_trend[test_col],
                            mode='lines+markers',
                            name=test_col
                        ))
                    
                    fig.update_layout(
                        title='Daily Orders per Audience Member',
                        xaxis_title='Date',
                        yaxis_title='Orders per Audience Member',
                        legend_title='Group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data to display daily trend")
            else:
                st.info("Start date not defined for this cohort")
                
            # Additional insights section
            st.subheader("Additional Insights")
            
            with st.expander("Analysis Methodology"):
                st.markdown("""
                ### Incremental Impact Calculation Methodology
                
                The incremental impact is calculated using the following approach:
                
                1. **Audience Split**: Test group represents 70% of the audience, while control group represents 30%
                
                2. **Scaling Formula**: To compare test and control on equal footing, we scale both to 100% using:
                   - Test (scaled to 100%) = (Test metric * 100) / 70
                   - Control (scaled to 100%) = (Control metric * 100) / 30
                
                3. **Incremental Impact**: Incremental impact = Test (scaled) - Control (scaled)
                
                4. **Lift Percentage**: For per-audience metrics, lift is calculated as:
                   - Lift % = ((Test metric per audience - Control metric per audience) / Control metric per audience) * 100
                """)
            
            with st.expander("Test Group Setup"):
                st.markdown(f"""
                ### Test Group Information
                
                - **Cohort**: {cohort_descriptions.get(selected_cohort, selected_cohort)}
                - **Test Start Date**: {test_start_dates.get(selected_cohort).strftime('%B %d, %Y')}
                - **Test Duration**: {(pd.Timestamp.now() - test_start_dates.get(selected_cohort)).days} days
                """)
            
            with st.expander("Recommendations"):
                st.markdown("""
                ### Recommendations Based on Analysis
                
                Based on the incremental impact analysis, consider the following recommendations:
                
                1. **Audience Targeting**: Focus on segments showing highest lift percentages
                
                2. **Campaign Optimization**: Scale successful campaigns with positive ROI
                
                3. **Recency Strategy**: Tailor messaging based on recency performance
                
                4. **Further Testing**: Consider A/B testing variations to improve performance
                """)
        else:
            st.info(f"No data available for {selected_cohort} with the selected filters")
    else:
        st.info("Please select a cohort from the sidebar to view detailed analysis")
