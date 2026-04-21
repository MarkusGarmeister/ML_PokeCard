Pokemon Card Rarity Classifier

This is a machine learning project that classifies Pokemon trading cards by their rarity.

Setup
Python 3.10 or higher is needed
Install all the dependecies
uv sync

Type of ML
Supervised, single-label multi-class image classification
Input: (120, 168, 3)
Output: 5 classes: ["Common", "Uncommon", "Rare", "Ultra Rare", "Secret, Rare"]

Sucess metrics:

Benchmarks:
Random Guessing - predicts always a random class, has an accuracy of 20% 1/5 classes
Majority Guessing - predicts always the most frequent class
