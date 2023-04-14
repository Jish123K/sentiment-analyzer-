import bert

import distilbert

import spacy

import huggingface_transformer

import vedint_sedimentatar

import textblob

# Load the pretrained model

bert_model = bert.BertModel()

distilbert_model = distilbert.DistilBertModel()

spacy_model = spacy.load("en_core_web_sm")

huggingface_transformer_model = huggingface_transformer.TransformerModel("bert-base-uncased")

vedint_sedimentatar_model = vedint_sedimentatar.SedimentatarModel()

textblob_model = textblob.TextBlob()

# Read the product reviews

reviews = []

with open("product_reviews.txt", "r") as f:

    for line in f:

        reviews.append(line)

# Perform sentiment analysis on the product reviews

sentiments = []

for review in reviews:

    # Use the pretrained model to predict the sentiment of the review

    sentiment = bert_model.predict(review)

    sentiments.append(sentiment)

# Print the overall sentiment of the reviews

print("Overall sentiment:", sentiments.count("positive") / len(sentiments))

# Print the aspects of the product that customers are happy or unhappy about

for sentiment, review in zip(sentiments, reviews):

    if sentiment == "positive":

        print("Customer is happy about:", review)

    else:

        print("Customer is unhappy about:", review)

# Extract the aspects of the product that customers are happy or unhappy about

happy_aspects = []

unhappy_aspects = []

for sentiment, review in zip(sentiments, reviews):

    if sentiment == "positive":

        happy_aspects.append(review)

    else:
      unhappy_aspects.append(review)

# Print the aspects of the product that customers are happy or unhappy about

print("Happy aspects:", happy_aspects)

print("Unhappy aspects:", unhappy_aspects)

# Print the most common happy and unhappy aspects

most_common_happy_aspects = Counter(happy_aspects).most_common(10)

most_common_unhappy_aspects = Counter(unhappy_aspects).most_common(10)

print("Most common happy aspects:", most_common_happy_aspects)

print("Most common unhappy aspects:", most_common_unhappy_aspects)

# Print the most common happy and unhappy aspects as a table

print("| Happy Aspects | Unhappy Aspects |")

print("|---|---|")

for happy_aspect, count in most_common_happy_aspects:

    print("|", happy_aspect, "|", count, "|")

for unhappy_aspect, count in most_common_unhappy_aspects:

    print("|", unhappy_aspect, "|", count, "|")

# Create a web application

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")

def index():

    return render_template("index.html",

                        overall_sentiment=sentiments.count("positive") / len(sentiments),

                        happy_aspects=happy_aspects,

                        unhappy_aspects=unhappy_aspects,

                        most_common_happy_aspects=most_common_happy_aspects,

                        most_common_unhappy_aspects=most_common_unhappy_aspects)
  if __name__ == "__main__":

    app.run(debug=True)
