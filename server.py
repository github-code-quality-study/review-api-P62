import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This is the list of allowed locations
        self.allowed_locations = [
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona"
        ]
        # This code preprocess reviews into a dictionary indexed by location to improve queries
        self.reviews_by_location = {}
        for review in reviews:
            location = review['Location']
            if location not in self.reviews_by_location:
               self.reviews_by_location[location] = []
            self.reviews_by_location[location].append(review)
        

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Extract query params from request
            query_string=environ.get('QUERY_STRING', '')
            params=parse_qs(query_string)

            location = params.get('location', [''])[0]
            start_date_param=params.get('start_date',[''])[0]
            end_date_param=params.get('end_date',[''])[0]

            # Convert params into datetime objects
            start_date = datetime.strptime(start_date_param, '%Y-%m-%d') if start_date_param else None
            end_date = datetime.strptime(end_date_param, '%Y-%m-%d') if end_date_param else None

            # Filter reviews by date and location
            filtered_reviews = []
            if location or start_date or end_date:
               if location in self.reviews_by_location:
                   for review in self.reviews_by_location[location]:
                       review_timestamp=datetime.strptime(review['Timestamp'],'%Y-%m-%d %H:%M:%S')
                       if ((not start_date or review_timestamp>=start_date) and 
                           (not end_date or review_timestamp<=end_date)):
                           filtered_reviews.append(review)
            else:
                filtered_reviews=reviews
            
            # Apply analyze_sentiment function to the filtered dict
            for review in filtered_reviews:
                review['sentiment']=self.analyze_sentiment(review['ReviewBody'])
            sorted_reviews=sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'],reverse=True)
            # Create the response body from the filtered_reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Extract query params from request
            try:
                content_length=int(environ.get('CONTENT_LENGTH', 0))
            except ValueError:
                content_length=0
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            review_data=parse_qs(post_data)
            review_body=review_data.get("ReviewBody", [''])[0]
            location=review_data.get("Location", [''])[0]

            # Handle exceptions when body is not correct
            if not review_body or not location:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "ReviewBody and Location are required."}).encode("utf-8")]
            if location not in self.allowed_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid location. The location must be one of the allowed locations."}).encode("utf-8")]

            # Create Timestamp and ReviewId
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            review_id=str(uuid.uuid4())

            # Create the new review
            new_review={
                "ReviewId":review_id,
                "Location":location,
                "Timestamp":timestamp,
                "ReviewBody":review_body,
            }
            # Add the new review to the corresponding dict
            reviews.append(new_review)
            if location not in self.reviews_by_location:
                self.reviews_by_location[location] = []
            self.reviews_by_location[location].append(new_review)

            # Create the response body from the filtered_reviews and convert to a JSON byte string
            response_body = json.dumps(new_review).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            return [response_body]
if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()