import json

file = r'E:\Documents\GitHub\learn\Books_small.json'
class Sentiment:
    negative = 'NEGATIVE'
    positive = 'POSITIVE'
    neutral = 'NEUTRAL'
    
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.negative
        elif self.score == 3:
            return Sentiment.neutral
        else: #score = 4 or 5
            return Sentiment.positive


reviews=[]
with open(file) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))
        
