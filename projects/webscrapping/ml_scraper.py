"""
Advanced ML Web Scraper Module
Contains enhanced functionality for web scraping with machine learning capabilities.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMLWebScraper:
    """
    An advanced web scraper that uses machine learning to enhance data extraction.
    
    This class provides capabilities for:
    - Web page classification using ML models
    - Structured data extraction from HTML content
    - Text preprocessing and feature engineering
    - Model training and evaluation
    """
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the scraper with specified model type.
        
        Args:
            model_type (str): Type of ML model to use ('naive_bayes', 'logistic_regression')
        """
        self.model_type = model_type
        self.classifier = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False
        
    def fetch_page(self, url, timeout=10):
        """
        Fetch HTML content from URL with error handling.
        
        Args:
            url (str): The URL to fetch
            timeout (int): Request timeout in seconds
            
        Returns:
            str or None: HTML content if successful, None otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, html_content):
        """
        Extract clean text content from HTML.
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            str: Cleaned text content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for ML processing.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower()
    
    def train_classifier(self, urls_and_labels):
        """
        Train a classifier to categorize web pages.
        
        Args:
            urls_and_labels (list): List of tuples (url, label) where label is category
        """
        logger.info("Training classifier...")
        
        # Extract URLs and labels
        urls = [item[0] for item in urls_and_labels]
        labels = [item[1] for item in urls_and_labels]
        
        # Fetch content from URLs
        contents = []
        valid_urls = []
        
        for url in urls:
            html_content = self.fetch_page(url)
            if html_content:
                text = self.extract_text(html_content)
                processed_text = self.preprocess_text(text)
                contents.append(processed_text)
                valid_urls.append(url)
            else:
                # Skip failed URLs
                logger.warning(f"Failed to fetch content from {url}")
                continue
        
        if len(contents) < 2:
            raise ValueError("Need at least two valid samples for training")
        
        # Train classifier based on model type
        if self.model_type == 'naive_bayes':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.classifier = pipeline.fit(contents, labels)
        self.is_trained = True
        
        logger.info("Training completed successfully")
    
    def classify_page(self, url):
        """
        Classify a web page using trained model.
        
        Args:
            url (str): URL to classify
            
        Returns:
            dict or None: Classification result with confidence score
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
            
        html_content = self.fetch_page(url)
        if not html_content:
            return None
            
        text = self.extract_text(html_content)
        processed_text = self.preprocess_text(text)
        
        prediction = self.classifier.predict([processed_text])[0]
        probability = self.classifier.predict_proba([processed_text])[0].max()
        
        return {
            'url': url,
            'category': prediction,
            'confidence': float(probability)
        }
    
    def extract_structured_data(self, url):
        """
        Extract structured data from a web page.
        
        Args:
            url (str): URL to extract data from
            
        Returns:
            dict or None: Structured data including title and links
        """
        html_content = self.fetch_page(url)
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_element = soup.find('title')
        title = title_element.get_text() if title_element else ''
        
        # Extract links with text and href attributes
        links = []
        for link in soup.find_all('a', href=True):
            link_data = {
                'text': link.get_text().strip(),
                'url': link['href']
            }
            # Filter out empty or very short texts
            if len(link_data['text']) > 2:
                links.append(link_data)
                
        return {
            'url': url,
            'title': title,
            'links': links[:10]  # Limit to first 10 links for brevity
        }
    
    def batch_classify(self, urls):
        """
        Classify multiple URLs in batch.
        
        Args:
            urls (list): List of URLs to classify
            
        Returns:
            list: List of classification results
        """
        results = []
        for url in urls:
            result = self.classify_page(url)
            if result:
                results.append(result)
            time.sleep(0.5)  # Be respectful to servers
        return results
    
    def batch_extract(self, urls):
        """
        Extract structured data from multiple URLs.
        
        Args:
            urls (list): List of URLs to extract data from
            
        Returns:
            list: List of extracted data dictionaries
        """
        results = []
        for url in urls:
            result = self.extract_structured_data(url)
            if result:
                results.append(result)
            time.sleep(0.5)  # Be respectful to servers
        return results

# Example usage function
def demo_usage():
    """Demonstrate the scraper functionality."""
    
    # Sample training data (in practice, you'd have more diverse examples)
    sample_data = [
        ("https://en.wikipedia.org/wiki/Machine_learning", "education"),
        ("https://www.python.org/", "technology"),
        ("https://stackoverflow.com/questions/tagged/python", "programming"),
        ("https://github.com/", "development"),
        ("https://news.ycombinator.com/", "technology")
    ]
    
    # Initialize scraper
    scraper = AdvancedMLWebScraper()
    
    print("Training classifier...")
    try:
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
                
    except Exception as e:
        logger.error(f"Error during demo: {e}")

if __name__ == "__main__":
    demo_usage()