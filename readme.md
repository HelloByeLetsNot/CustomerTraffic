# 🚦 Traffic Analyzer: AI-Powered Traffic Insights

## 🌟 Overview

Traffic Analyzer is an advanced data analysis application designed to help businesses and organizations gain deep insights into their traffic patterns using machine learning and AI-powered analytics.

![Traffic Analyzer Logo](logo.png)

## ✨ Key Features

### 🤖 AI-Powered Analysis
- Leverage OpenAI's GPT models for intelligent traffic data interpretation
- Generate human-readable insights and predictions
- Customize analysis prompts for specific business needs

### 📊 Machine Learning Predictions
- Train predictive models using Random Forest Regression
- Forecast traffic trends based on historical data
- Identify key factors influencing traffic patterns

### 💾 Persistent Data Management
- SQLite database integration
- Store and track traffic data
- Log model performance metrics
- Maintain historical analysis records

### 📈 Comprehensive Visualization
- Upload and analyze CSV traffic data
- Support for complex traffic datasets
- Graphical representation of insights

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Methods

#### 1. PyPI Installation
```bash
pip install traffic-analyzer
```

#### 2. From Source
```bash
git clone https://github.com/yourusername/traffic-analyzer.git
cd traffic-analyzer
pip install -e .
```

## 🚀 Quick Start

### Running the Application
```bash
traffic-analyzer
```

### First-Time Setup
1. Open the application
2. Navigate to Settings → API Settings
3. Enter your OpenAI API Key
4. Upload a CSV file with traffic data
   - Required columns: 'Date', 'Traffic', 'Hour'

## 📝 Data Preparation

### CSV File Format
Your input CSV should have the following structure:
- `Date`: Timestamp of traffic measurement
- `Traffic`: Numeric traffic volume
- `Hour`: Hour of the day (0-23)

Example:
```csv
Date,Traffic,Hour
2023-06-01 14:30:00,150.5,14
2023-06-01 15:00:00,175.2,15
```

## 🧠 Machine Learning Workflow

1. **Upload Data**: Load your traffic CSV
2. **Train Model**: Click "Train ML Model"
   - Model learns from historical patterns
   - Outputs performance metrics (MSE, R2 Score)
3. **Make Predictions**: Use "Make Predictions"
   - Forecast traffic for specific time periods

## 🔍 AI Analysis Capabilities

- Identify peak traffic periods
- Detect seasonal trends
- Provide strategic business insights
- Customize analysis through prompt engineering

## 🛡 Configuration

### OpenAI API Key
- Obtain from [OpenAI Platform](https://platform.openai.com/)
- Keep key confidential
- Manage in application settings

### Supported GPT Models
- gpt-3.5-turbo (default)
- Configurable model selection (future version)

## 🔧 Troubleshooting

### Common Issues
- **Missing API Key**: Set in Settings menu
- **Incorrect CSV Format**: Verify column names
- **Prediction Errors**: Ensure sufficient training data

## 📊 Performance Metrics

The application tracks:
- Model Mean Squared Error (MSE)
- R² Score
- Training Timestamp

## 🤝 Contributing

### Ways to Contribute
- Report bugs
- Suggest features
- Submit pull requests

### Development Setup
```bash
git clone https://github.com/HelloByeLetsNot/CustomerTraffic.git
cd traffic-analyzer
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## 📄 License

MIT License - See LICENSE file for details

## 🏷️ Version

Current Version: 0.1.0

## 📧 Contact & Support

- **Email**: kodytryon@proton.me
- **Issues**: (https://github.com/HelloByeLetsNot/CustomerTraffic)

## 🌐 Follow Us
- Twitter: @TrafficAnalyzer
- LinkedIn: Traffic Analyzer Community

---

**Disclaimer**: Traffic predictions are based on historical data and machine learning models. Always combine AI insights with human expertise for critical business decisions.