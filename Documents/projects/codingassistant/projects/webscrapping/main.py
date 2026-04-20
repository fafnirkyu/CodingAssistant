"""
ML Web Scraper - Main Entry Point
This script demonstrates an ML-enhanced web scraper that classifies web pages 
and extracts structured data using machine learning techniques.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import time

# Define a simple web scraper with ML capabilities
class MLWebScraper:
    def __init__(self):
        self.classifier = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def fetch_page(self, url):
        """Fetch HTML content from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, html_content):
        """Extract text content from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower()
    
    def train_classifier(self, urls_and_labels):
        """
        Train a classifier to categorize web pages
        
        Args:
            urls_and_labels: List of tuples (url, label) where label is category
        """
        # Extract URLs and labels
        urls = [item[0] for item in urls_and_labels]
        labels = [item[1] for item in urls_and_labels]
        
        # Fetch content from URLs
        contents = []
        for url in urls:
            html_content = self.fetch_page(url)
            if html_content:
                text = self.extract_text(html_content)
                processed_text = self.preprocess_text(text)
                contents.append(processed_text)
            else:
                contents.append("")
        
        # Train classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        self.classifier = pipeline.fit(contents, labels)
    
    def classify_page(self, url):
        """Classify a web page using trained model"""
        html_content = self.fetch_page(url)
        if not html_content:
            return None
            
        text = self.extract_text(html_content)
        processed_text = self.preprocess_text(text)
        
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
            
        prediction = self.classifier.predict([processed_text])[0]
        probability = self.classifier.predict_proba([processed_text])[0].max()
        
        return {
            'url': url,
            'category': prediction,
            'confidence': probability
        }
    
    def extract_structured_data(self, url):
        """Extract structured data from a web page"""
        html_content = self.fetch_page(url)
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Example: Extract title and all links
        title = soup.find('title').get_text() if soup.find('title') else ''
        
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                'text': link.get_text().strip(),
                'url': link['href']
            })
            
        return {
            'url': url,
            'title': title,
            'links': links[:10]  # Limit to first 10 links
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Sample training data (in practice, you'd have more diverse examples)
    sample_data = [
        ("https://en.wikipedia.org/wiki/Machine_learning", "education"),
        ("https://www.python.org/", "technology"),
        ("https://stackoverflow.com/questions/tagged/python", "programming"),
        ("https://github.com/", "development"),
        ("https://news.ycombinator.com/", "technology")
    ]
    
    # Initialize scraper
    scraper = MLWebScraper()
    
    print("Training classifier...")
    scraper.train_classifier(sample_data)
    
    print("\nTesting classification on new URLs:")
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://www.github.com/trending",
        "https://stackoverflow.com/questions/tagged/machine-learning"
    ]
    
    for url in test_urls:
        result = scraper.classify_page(url)
        if result:
            print(f"URL: {result['url']}")
            print(f"Category: {result['category']} (Confidence: {result['confidence']:.2f})")
        time.sleep(1)  # Be respectful to servers
    
    print("\nExtracting structured data:")
    for url in test_urls[:2]:
        result = scraper.extract_structured_data(url)
        if result:
            print(f"Title of {url}: {result['title']}")