# Fake-News-Detecting-System
This system aims to solve the breach in data authenticity, especially for local news and text messages
Fake-News-Detecting-System
📰 Fake News Detector
This is a Flask-based web app that detects whether a news article is fake or real using:

Machine Learning (Logistic Regression)
Entity Extraction & Fact-checking
Wikipedia-based Verification
News API Cross-checking (optional)
Features
Paste any news content to verify.
Displays prediction + confidence + insights.
Logs all user checks.
Deployment
This app is deployed on Render using:

interface/app.py as the entry point
gunicorn to serve the app
requirements.txt for dependencies
Folder Structure
fake-news-detecting/ ├── interface/ ← Flask app lives here ├── model/ ← ML model stored here ├── data/ ← News dataset ├── utils/ ← Helper functions ├── verifier.py ← Final decision logic ├── analytics.py ← Usage insights ├── requirements.txt ├── README.md

How to Use
Go to the deployed Render URL.
Paste a news article.
Click "Verify".
Get the result instantly.
