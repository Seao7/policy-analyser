import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
import google.generativeai as genai
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import google.generativeai as genai

# At the top of your file, after imports
def configure_gemini():
    """Configure Gemini API key from Streamlit secrets"""
    try:
        # Try to get from Streamlit secrets first
        api_key = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=api_key)
        return True
    except:
        return False

# Configure the page
st.set_page_config(
    page_title="Policy Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace the sidebar API configuration section with:
st.sidebar.header("🔐 API Configuration")

# Try to configure automatically from secrets
if configure_gemini():
    st.sidebar.success("✅ Gemini API configured from secrets")
    gemini_configured = True
else:
    # Fallback to manual input
    st.sidebar.info("Enter your Gemini API key manually:")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_configured = True
    else:
        gemini_configured = False

# Title and description
st.title("🏦 LIC Policy Analyzer")
st.markdown("### Analyze if your LIC policy is Good or Bad")
st.markdown("---")

# Main input section
st.header("📊 Policy Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Premium Information")
    premium_amount = st.number_input(
        "Annual Premium Amount (₹)", 
        min_value=0.0, 
        value=50000.0, 
        step=1000.0,
        help="💡 The amount you pay every year during the premium payment period"
    )
    premium_years = st.number_input(
        "Premium Payment Period (Years)", 
        min_value=1, 
        value=15, 
        step=1,
        help="💡 Number of years you need to pay the premium"
    )
    
    st.subheader("Benefits Information")
    annual_payment = st.number_input(
        "Annual Payment After Premium Period (₹)", 
        min_value=0.0, 
        value=25000.0, 
        step=1000.0,
        help="💡 The yearly amount you'll receive after completing premium payments"
    )
    basic_sum_assured = st.number_input(
        "Basic Sum Assured (₹)", 
        min_value=0.0, 
        value=500000.0, 
        step=10000.0,
        help="💡 The base death benefit amount mentioned in your policy"
    )
    bonus_multiple = st.number_input(
        "Bonus Multiple", 
        min_value=1.0, 
        value=2.5, 
        step=0.1,
        help="💡 The multiplier applied to basic sum assured (including bonuses). Usually 2-3x"
    )

with col2:
    st.subheader("Analysis Parameters")
    life_expectancy = st.number_input(
        "Expected Life Span (Years from now)", 
        min_value=premium_years, 
        value=60, 
        step=1,
        help="💡 How many more years you expect to live from today"
    )
    annual_payment_years = st.number_input(
        "Years to Receive Annual Payments", 
        min_value=1, 
        value=life_expectancy-premium_years, 
        step=1,
        help="💡 Number of years you'll receive annual payments (usually Life Expectancy - Premium Years)"
    )
    comparison_rate = st.number_input(
        "Comparison Interest Rate (%)", 
        min_value=0.0, 
        value=8.0, 
        step=0.1,
        help="💡 Interest rate you could earn from alternative investments (FD, mutual funds, etc.) - this should already include your inflation expectations"
    ) / 100

# Analysis functions
def calculate_irr(premium_amount, premium_years, annual_payment, annual_payment_years, death_benefit):
    """Calculate Internal Rate of Return (IRR) for the policy"""
    
    def npv_function(rate):
        # Cash outflows (premiums)
        cash_outflows = -premium_amount * ((1 - (1 + rate)**(-premium_years)) / rate) if rate != 0 else -premium_amount * premium_years
        
        # Cash inflows (annual payments)
        if rate != 0:
            pv_annual_payments = annual_payment * ((1 - (1 + rate)**(-(annual_payment_years))) / rate) * (1 + rate)**(-premium_years)
        else:
            pv_annual_payments = annual_payment * annual_payment_years * (1 + rate)**(-premium_years)
        
        # Death benefit at the end
        pv_death_benefit = death_benefit * (1 + rate)**(-(premium_years + annual_payment_years))
        
        return cash_outflows + pv_annual_payments + pv_death_benefit
    
    try:
        irr = fsolve(npv_function, 0.05)[0]
        return irr
    except:
        return None

def calculate_pv_analysis(premium_amount, premium_years, annual_payment, annual_payment_years, 
                         death_benefit, discount_rate):
    """Calculate Present Value analysis - clean version without inflation double-counting"""
    
    # Present Value of premiums paid
    if discount_rate != 0:
        pv_premiums = premium_amount * ((1 - (1 + discount_rate)**(-premium_years)) / discount_rate)
    else:
        pv_premiums = premium_amount * premium_years
    
    # Present Value of annual payments
    if discount_rate != 0:
        pv_annual_payments = annual_payment * ((1 - (1 + discount_rate)**(-(annual_payment_years))) / discount_rate) * (1 + discount_rate)**(-premium_years)
    else:
        pv_annual_payments = annual_payment * annual_payment_years * (1 + discount_rate)**(-premium_years)
    
    # Present Value of death benefit
    pv_death_benefit = death_benefit * (1 + discount_rate)**(-(premium_years + annual_payment_years))
    
    # Net Present Value
    npv = pv_annual_payments + pv_death_benefit - pv_premiums
    
    return {
        'pv_premiums': pv_premiums,
        'pv_annual_payments': pv_annual_payments,
        'pv_death_benefit': pv_death_benefit,
        'total_pv_benefits': pv_annual_payments + pv_death_benefit,
        'npv': npv,
        'benefit_cost_ratio': (pv_annual_payments + pv_death_benefit) / pv_premiums
    }

def generate_simplified_explanation(premium_amount, premium_years, annual_payment, annual_payment_years, 
                                  death_benefit, comparison_rate, pv_analysis):
    """Generate simplified explanation using user's actual values"""
    
    total_premiums_paid = premium_amount * premium_years
    total_annual_payments = annual_payment * annual_payment_years
    
    explanation = f"""
    ## 💡 Understanding Your Policy in Simple Terms
    
    ### The Basic Concept: Money Today vs Money Tomorrow
    
    **Simple Question**: Would you rather have ₹100 today or ₹100 after 5 years?
    
    **Answer**: ₹100 today! Because you can invest that ₹100 today and it will grow.
    
    If you invest ₹100 today at {comparison_rate*100:.1f}% per year, it becomes ₹{100 * (1 + comparison_rate)**5:,.0f} in 5 years.
    
    So ₹100 today = ₹{100 * (1 + comparison_rate)**5:,.0f} in 5 years.
    
    **Reverse calculation**: ₹{100 * (1 + comparison_rate)**5:,.0f} received in 5 years = ₹100 in today's money.
    
    This "today's money value" is called **Present Value**.
    
    ---
    
    ## 📊 Your LIC Policy Analysis
    
    ### Your Policy Details:
    - **Premium**: ₹{premium_amount:,.0f} per year for {premium_years} years
    - **Annual payments**: ₹{annual_payment:,.0f} per year for {annual_payment_years} years (starting after premium period)
    - **Death benefit**: ₹{death_benefit:,.0f} (at the end)
    - **Your comparison rate**: {comparison_rate*100:.1f}% (includes your inflation expectations)
    
    ### Timeline:
    ```
    Years 1-{premium_years}:     You PAY ₹{premium_amount:,.0f} each year
    Years {premium_years+1}-{premium_years + annual_payment_years}: You GET ₹{annual_payment:,.0f} each year  
    Year {premium_years + annual_payment_years + 1}:      You GET ₹{death_benefit:,.0f} (death benefit)
    ```
    
    ---
    
    ## 🧮 Step-by-Step Present Value Calculation
    
    ### Step 1: Present Value of What You Pay (Premiums)
    
    You pay ₹{premium_amount:,.0f} for {premium_years} years. Converting to today's money using {comparison_rate*100:.1f}% rate:
    
    **Total Present Value of Premiums = ₹{pv_analysis['pv_premiums']:,.0f}**
    
    ### Step 2: Present Value of Annual Payments
    
    You get ₹{annual_payment:,.0f} for {annual_payment_years} years, but starting from Year {premium_years+1}:
    
    **Present Value of Annual Payments = ₹{pv_analysis['pv_annual_payments']:,.0f}**
    
    ### Step 3: Present Value of Death Benefit
    
    ₹{death_benefit:,.0f} received in Year {premium_years + annual_payment_years + 1}:
    
    **Present Value of Death Benefit = ₹{pv_analysis['pv_death_benefit']:,.0f}**
    
    ---
    
    ## 📈 Final Calculation
    
    ### Total Benefits (in today's money):
    - Annual payments: ₹{pv_analysis['pv_annual_payments']:,.0f}
    - Death benefit: ₹{pv_analysis['pv_death_benefit']:,.0f}
    - **Total Benefits = ₹{pv_analysis['total_pv_benefits']:,.0f}**
    
    ### Total Costs (in today's money):
    - Premiums: ₹{pv_analysis['pv_premiums']:,.0f}
    
    ### Result:
    - **Net Present Value** = ₹{pv_analysis['total_pv_benefits']:,.0f} - ₹{pv_analysis['pv_premiums']:,.0f} = **₹{pv_analysis['npv']:,.0f}**
    
    ---
    
    ## 🎯 What This Means in Simple Terms
    
    **If you took that same ₹{premium_amount:,.0f} per year and invested it at {comparison_rate*100:.1f}%:**
    """
    
    if pv_analysis['npv'] > 0:
        explanation += f"""
        - You'd end up with **less** money than what the LIC policy gives you
        - The policy is **GOOD** because NPV is positive (₹{pv_analysis['npv']:,.0f})
        - You're **gaining** ₹{abs(pv_analysis['npv']):,.0f} in today's purchasing power! 🎉
        """
    else:
        explanation += f"""
        - You'd end up with **more** money than what the LIC policy gives you
        - The policy is **BAD** because NPV is negative (₹{pv_analysis['npv']:,.0f})
        - You're **losing** ₹{abs(pv_analysis['npv']):,.0f} in today's purchasing power! ⚠️
        """
    
    explanation += f"""
    
    ## 🔍 Key Takeaway
    
    The calculator is simply answering: **"If I invest the same premium amount at {comparison_rate*100:.1f}%, will I get more or less money than this LIC policy?"**
    
    - **Positive NPV** = LIC policy is better than your comparison investment
    - **Negative NPV** = Your comparison investment is better than LIC
    
    **Your result: NPV = ₹{pv_analysis['npv']:,.0f}**
    
    That's it! The math is just a tool to make this comparison accurately.
    
    ---
    
    ## 📚 Why We Don't Separately Adjust for Inflation
    
    Your {comparison_rate*100:.1f}% comparison rate should already include:
    - **Real return expectations** (2-3%)
    - **Inflation expectations** (4-6%)
    - **Risk premium** (if applicable)
    
    So when we discount future cash flows at {comparison_rate*100:.1f}%, we're already accounting for inflation!
    """
    
    return explanation

def get_llm_analysis(policy_data, irr, pv_analysis):
    """Get LLM analysis using Gemini"""
    api_key = st.secrets["gemini"]["api_key"]
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
    Analyze this LIC policy and provide comprehensive guidance:
    
    Policy Details:
    - Annual Premium: ₹{policy_data['premium']:,.2f} for {policy_data['premium_years']} years
    - Total Premiums Paid: ₹{policy_data['premium'] * policy_data['premium_years']:,.2f}
    - Annual Payment: ₹{policy_data['annual_payment']:,.2f} for {policy_data['annual_payment_years']} years
    - Death Benefit: ₹{policy_data['death_benefit']:,.2f}
    
    Financial Analysis:
    - Effective Interest Rate (IRR): {irr*100:.2f}% per annum
    - Present Value of Premiums: ₹{pv_analysis['pv_premiums']:,.2f}
    - Present Value of Benefits: ₹{pv_analysis['total_pv_benefits']:,.2f}
    - Net Present Value: ₹{pv_analysis['npv']:,.2f}
    - Benefit-Cost Ratio: {pv_analysis['benefit_cost_ratio']:.2f}
    - Comparison Rate: {policy_data['comparison_rate']*100:.1f}%
    
    Please provide:
    1. Overall assessment (Good/Bad/Average)
    2. Key strengths and weaknesses
    3. Comparison with alternative investments
    4. Specific recommendations
    5. Action items for the policy holder
    
    Keep the analysis practical and actionable for someone making financial decisions.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Perform calculations when button is clicked
if st.button("🔍 Analyze Policy", type="primary"):
    if premium_amount > 0 and premium_years > 0:
        
        # Calculate death benefit
        death_benefit = basic_sum_assured * bonus_multiple
        
        # Calculate IRR
        irr = calculate_irr(premium_amount, premium_years, annual_payment, 
                           annual_payment_years, death_benefit)
        
        # Calculate PV Analysis (without inflation parameter)
        pv_analysis = calculate_pv_analysis(premium_amount, premium_years, annual_payment,
                                           annual_payment_years, death_benefit, 
                                           comparison_rate)
        
        # Display results
        st.header("📈 Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if irr:
                st.metric("Effective Interest Rate", f"{irr*100:.2f}%")
            else:
                st.metric("Effective Interest Rate", "Unable to calculate")
        
        with col2:
            st.metric("Net Present Value", f"₹{pv_analysis['npv']:,.0f}")
        
        with col3:
            st.metric("Benefit-Cost Ratio", f"{pv_analysis['benefit_cost_ratio']:.2f}")
        
        with col4:
            st.metric("Total Death Benefit", f"₹{death_benefit:,.0f}")
        
        # Add the simplified explanation as an expandable section
        with st.expander("🧠 **Want to understand HOW these numbers are calculated? Click here for simple explanation!**", expanded=False):
            explanation = generate_simplified_explanation(
                premium_amount, premium_years, annual_payment, annual_payment_years,
                death_benefit, comparison_rate, pv_analysis
            )
            st.markdown(explanation)
        
        # Detailed breakdown
        st.subheader("💰 Financial Breakdown")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.write("**Costs (Present Value)**")
            st.write(f"• Total Premiums: ₹{pv_analysis['pv_premiums']:,.2f}")
            
            st.write("**Benefits (Present Value)**")
            st.write(f"• Annual Payments: ₹{pv_analysis['pv_annual_payments']:,.2f}")
            st.write(f"• Death Benefit: ₹{pv_analysis['pv_death_benefit']:,.2f}")
            st.write(f"• **Total Benefits: ₹{pv_analysis['total_pv_benefits']:,.2f}**")
        
        with breakdown_col2:
            # Create a simple comparison chart
            fig = go.Figure()
            
            categories = ['Premiums Paid', 'Benefits Received']
            values = [pv_analysis['pv_premiums'], pv_analysis['total_pv_benefits']]
            colors = ['red', 'green']
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"₹{v:,.0f}" for v in values],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Present Value Comparison",
                yaxis_title="Amount (₹)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Scenario Analysis
        st.subheader("📊 Scenario Analysis")
        
        scenarios = []
        rates = [0.06, 0.08, 0.10, 0.12]
        
        for rate in rates:
            scenario_pv = calculate_pv_analysis(premium_amount, premium_years, annual_payment,
                                              annual_payment_years, death_benefit, rate)
            scenarios.append({
                'Interest Rate': f"{rate*100:.0f}%",
                'NPV': scenario_pv['npv'],
                'Benefit-Cost Ratio': scenario_pv['benefit_cost_ratio']
            })
        
        scenario_df = pd.DataFrame(scenarios)
        st.dataframe(scenario_df, use_container_width=True)
        
        # Cash flow timeline
        st.subheader("💸 Cash Flow Timeline")
        
        years = list(range(1, premium_years + annual_payment_years + 2))
        cash_flows = []
        
        for year in years:
            if year <= premium_years:
                cash_flows.append(-premium_amount)
            elif year <= premium_years + annual_payment_years:
                cash_flows.append(annual_payment)
            else:
                cash_flows.append(death_benefit)
        
        timeline_fig = go.Figure()
        colors = ['red' if cf < 0 else 'green' for cf in cash_flows]
        
        timeline_fig.add_trace(go.Bar(
            x=years,
            y=cash_flows,
            marker_color=colors,
            text=[f"₹{abs(cf):,.0f}" for cf in cash_flows],
            textposition='auto',
        ))
        
        timeline_fig.update_layout(
            title="Cash Flow Timeline",
            xaxis_title="Year",
            yaxis_title="Cash Flow (₹)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # AI Analysis
        st.header("🤖 AI Analysis & Recommendations")
        
        if gemini_configured:
            with st.spinner("Generating AI analysis..."):
                policy_data = {
                    'premium': premium_amount,
                    'premium_years': premium_years,
                    'annual_payment': annual_payment,
                    'annual_payment_years': annual_payment_years,
                    'death_benefit': death_benefit,
                    'comparison_rate': comparison_rate
                }
                
                ai_analysis = get_llm_analysis(policy_data, irr, pv_analysis)
                st.write(ai_analysis)
        else:
            st.warning("Configure your Gemini API key to get AI-powered analysis.")

            
            # Provide basic analysis
            st.subheader("Basic Analysis")
            
            if irr and irr > comparison_rate:
                st.success(f"✅ **Good Policy**: The effective interest rate ({irr*100:.2f}%) is higher than your comparison rate ({comparison_rate*100:.1f}%)")
            elif irr and irr < comparison_rate:
                st.error(f"❌ **Poor Policy**: The effective interest rate ({irr*100:.2f}%) is lower than your comparison rate ({comparison_rate*100:.1f}%)")
            else:
                st.info("ℹ️ **Average Policy**: The returns are comparable to your comparison rate")
            
            if pv_analysis['npv'] > 0:
                st.success(f"✅ **Positive NPV**: ₹{pv_analysis['npv']:,.0f}")
            else:
                st.error(f"❌ **Negative NPV**: ₹{pv_analysis['npv']:,.0f}")
    
    else:
        st.error("Please enter valid premium amount and premium years.")

# Additional Information
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    ### Instructions:
    1. **Enter Policy Details**: Fill in your premium amount, payment period, and benefit details
    2. **Set Analysis Parameters**: Enter your life expectancy and comparison interest rate
    3. **Optional**: Add your Gemini API key for AI-powered analysis
    4. **Analyze**: Click the "Analyze Policy" button to get comprehensive results
    
    ### Key Metrics Explained:
    - **Effective Interest Rate (IRR)**: The actual return rate your policy provides
    - **Net Present Value (NPV)**: Difference between present value of benefits and premiums
    - **Benefit-Cost Ratio**: How much benefit you get per rupee of premium paid
    
    ### About the Comparison Rate:
    - Choose a rate that represents your best alternative investment option
    - This rate should already include your inflation expectations
    - For example: If you expect 6% inflation and want 2% real returns, use 8% as comparison rate
    
    ### Interpretation:
    - **Good Policy**: IRR > Comparison rate AND NPV > 0
    - **Poor Policy**: IRR < Comparison rate AND NPV < 0
    - **Average Policy**: IRR ≈ Comparison rate
    """)

st.markdown("---")
st.markdown("**Disclaimer**: This analysis is for educational purposes only. Please consult with a financial advisor for investment decisions.")
