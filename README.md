# 🏦 Policy Analyzer

A comprehensive Streamlit application to analyze policies and determine if they're financially beneficial compared to alternative investments.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)


## 🎯 Overview

The LIC Policy Analyzer helps you make informed decisions about your Life Insurance policies by:

- **Calculating the effective interest rate (IRR)** your policy provides
- **Comparing with alternative investments** using Present Value analysis
- **Providing AI-powered insights** and recommendations via Google Gemini
- **Visualizing cash flows** over the policy timeline
- **Educational explanations** of financial concepts in simple terms


## ✨ Features

### 📊 **Financial Analysis**

- **Internal Rate of Return (IRR)** calculation using numerical methods
- **Net Present Value (NPV)** analysis with proper time value of money calculations
- **Benefit-Cost Ratio** evaluation
- **Scenario analysis** across different interest rates (6%, 8%, 10%, 12%)


### 📈 **Interactive Visualizations**

- Present value comparison charts
- Cash flow timeline visualization
- Scenario analysis tables
- Interactive Plotly charts with hover details


### 🤖 **AI-Powered Insights**

- Integration with Google Gemini AI for intelligent analysis
- Comprehensive policy assessment with pros/cons
- Personalized recommendations based on your specific numbers
- Alternative investment comparisons and action items


### 🧠 **Educational Component**

- **Expandable explanations** of complex financial concepts
- **Step-by-step calculation breakdowns** using your actual policy data
- **Time value of money education** with real examples
- **Clear interpretation** of whether your policy is good or bad


### 🔧 **User-Friendly Interface**

- Clean, intuitive Streamlit interface
- **Helpful tooltips** (💡) for all input fields
- Expandable sections for detailed explanations
- Responsive design that works on desktop and mobile


## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager


### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/lic-policy-analyzer.git
cd lic-policy-analyzer
```


### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv lic_analyzer_env
source lic_analyzer_env/bin/activate  # On Windows: lic_analyzer_env\Scripts\activate
```


### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```


### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## ⚙️ Configuration

### API Key Setup (Optional but Recommended)

To enable AI-powered analysis, you need a Google Gemini API key. The application supports multiple configuration methods:

#### Method 1: Streamlit Secrets (Recommended for Production)

1. Create `.streamlit/secrets.toml` in your project root:
```toml
[gemini]
api_key = "your_gemini_api_key_here"
```


#### Method 2: Environment Variables

1. Install python-dotenv: `pip install python-dotenv`
2. Create `.env` file in your project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```


#### Method 3: Manual Input

- Simply enter your API key in the sidebar when using the app


### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy and use it in your preferred configuration method

**Note**: The app works without an API key, but you'll miss the AI-powered insights and recommendations.

## 🚀 Usage

### Input Parameters

#### **Premium Information**

- **Annual Premium Amount**: The amount you pay every year
- **Premium Payment Period**: Number of years you pay premiums


#### **Benefits Information**

- **Annual Payment After Premium Period**: Yearly amount received after paying premiums
- **Basic Sum Assured**: Base death benefit mentioned in policy
- **Bonus Multiple**: Multiplier applied to sum assured (typically 2-3x)


#### **Analysis Parameters**

- **Expected Life Span**: Years you expect to live from today
- **Years to Receive Annual Payments**: Usually (Life Expectancy - Premium Years)
- **Comparison Interest Rate**: Rate you could earn from alternatives (FD, mutual funds, etc.)


### Understanding Results

#### **Key Metrics**

- **Effective Interest Rate (IRR)**: The actual annual return your policy provides
- **Net Present Value (NPV)**:
    - **Positive NPV**: Policy is better than your comparison investment ✅
    - **Negative NPV**: Alternative investments are better ❌
- **Benefit-Cost Ratio**: Benefit per rupee of premium paid


#### **Policy Assessment**

- **Good Policy**: IRR > Comparison rate AND NPV > 0
- **Poor Policy**: IRR < Comparison rate AND NPV < 0
- **Average Policy**: IRR ≈ Comparison rate


### Advanced Features

#### **Educational Section**

Click "🧠 Want to understand HOW these numbers are calculated?" for:

- Simple explanation of time value of money
- Step-by-step present value calculations using your numbers
- Clear interpretation of results
- Why inflation is already accounted for in your comparison rate


#### **Scenario Analysis**

Automatically shows how your policy performs at different interest rates

#### **Cash Flow Timeline**

Visual representation of your payment schedule and benefits

## 🔍 How It Works

### Core Financial Concepts

#### **Present Value Calculation**

```python
PV = Future_Value / (1 + interest_rate)^years
```


#### **Internal Rate of Return (IRR)**

The interest rate that makes NPV = 0:

```python
NPV = Σ(Cash_Flows / (1 + IRR)^t) = 0
```


#### **Net Present Value**

```python
NPV = PV(All_Benefits) - PV(All_Premiums)
```


### Analysis Process

1. **Calculate present value** of all premium payments
2. **Calculate present value** of annual benefits (received after premium period)
3. **Calculate present value** of death benefit
4. **Compare total benefits vs total costs** in today's money
5. **Find the effective interest rate** that the policy provides
6. **Generate AI-powered recommendations** based on analysis

### Why No Separate Inflation Adjustment?

The comparison interest rate you enter should already include your inflation expectations. For example:

- Real return expectation: 2%
- Inflation expectation: 6%
- **Total comparison rate: 8%**

This avoids double-counting inflation in the calculations.

## 📁 File Structure

```
lic-policy-analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── .gitignore           # Git ignore rules
├── .env.example         # Environment variables example
│
├── .streamlit/
│   └── secrets.toml.example  # Streamlit secrets example
│
└── tests/
    └── test_calculations.py  # Unit tests (future)
```


## 📦 Dependencies

```txt
streamlit
numpy
pandas
matplotlib
seaborn
scipy
google-generativeai
plotly
python-dotenv
```

For specific versions, see `requirements.txt`.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional policy types** (term, endowment, ULIP)
- **More sophisticated models** (tax considerations, riders)
- **Enhanced visualizations**
- **Unit tests** and validation
- **Performance optimizations**
- **UI/UX improvements**


### Development Setup

```bash
git clone https://github.com/yourusername/lic-policy-analyzer.git
cd lic-policy-analyzer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```


## 🔒 Security Notes

### **API Key Security**

- Never commit API keys to version control
- Use `.env` files for local development
- Use Streamlit secrets for deployment
- Add sensitive files to `.gitignore`


### **.gitignore Template**

```gitignore
# API Keys and secrets
.env
.streamlit/secrets.toml
*.key

# Python
__pycache__/
*.pyc
.venv/
```


## 🚀 Deployment

### **Streamlit Cloud**

1. Push code to GitHub (excluding secrets)
2. Connect repository on [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in app settings:
```toml
[gemini]
api_key = "your_api_key_here"
```


### **Local Production**

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```


## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

**This analysis is for educational purposes only.**

- **Not Financial Advice**: Results should not replace professional financial consultation
- **Simplified Model**: Real LIC policies have complex terms not captured here
- **Assumptions Required**: Analysis depends on assumptions about future performance
- **Consult Experts**: Always discuss with qualified financial advisors before major decisions


## 🆘 Support \& Issues

- **Bug Reports**: Open an issue on GitHub with detailed description
- **Feature Requests**: Describe your use case and proposed solution
- **Questions**: Check existing issues or create a new discussion


## 🎉 Acknowledgments

- **Streamlit** for the amazing web app framework
- **Google Gemini** for AI-powered insights
- **Plotly** for interactive visualizations
- **SciPy** for numerical computations
- The financial modeling community for best practices

**Made with ❤️ to help people make better financial decisions**

*Star ⭐ this repository if you found it helpful!*

