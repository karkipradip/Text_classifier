# Text Classification System

This project is a small text classification system that categorizes user messages into three classes:

- `Complaint`
- `Feedback`
- `Inquiry`

The classifier is built with Python and scikit-learn. It uses TF-IDF vectorization to convert text into numerical features and a Multinomial Naive Bayes model to predict the category of a message.

## Project Description

The goal of this project is to classify short user messages such as customer comments, questions, and negative reports into one of three labels:

- `Complaint`: negative experiences, problems, or dissatisfaction
- `Feedback`: opinions, suggestions, praise, or general comments
- `Inquiry`: questions or requests for information

This project is useful as a simple example of a natural language processing pipeline for message classification.

## Approach And Methodology

The system follows these steps:

1. Load labeled messages from a CSV dataset.
2. Preprocess the text by:
   - converting to lowercase
   - removing punctuation
   - normalizing extra spaces
   - correcting a few common typos
3. Split the dataset into training and testing sets using `train_test_split`.
4. Convert the training text into TF-IDF features.
5. Train a `MultinomialNB` classifier.
6. Evaluate the model using the test set and report accuracy.
7. Predict the category of new user input.

The current implementation also includes a small rule-based fallback for some short or typo-heavy messages, which helps improve practical interactive predictions.

## Dataset Details

The dataset is stored in:

- `data/messages.csv`

It contains labeled examples for all three classes:

- `Complaint`: 10 examples
- `Feedback`: 10 examples
- `Inquiry`: 16 examples

Each row has two columns:

```csv
text,label
"My order arrived damaged and nobody has responded yet.",Complaint
"Your latest update made the website much faster.",Feedback
"How can I reset my password?",Inquiry
```

## Files

- `text_classifier.py`: main classifier script
- `data/messages.csv`: labeled dataset

## How To Run

From the project root:

```powershell
python .\Classifier\text_classifier.py
```

This will:

- load the dataset
- split it into training and test sets
- train the classifier
- print accuracy and a classification report
- start interactive mode for manual testing

To classify one message directly:

```powershell
python .\Classifier\text_classifier.py "when will i get refund"
```

If `python` does not work on your machine, try:

```powershell
py .\Classifier\text_classifier.py
```

## Sample Inputs And Outputs

Example 1:

```text
Enter a message: not working
Predicted category: Complaint
```

Example 2:

```text
Enter a message: the new product is good
Predicted category: Feedback
```

Example 3:

```text
Enter a message: how do i change my billing address
Predicted category: Inquiry
```

Example direct command:

```powershell
python .\Classifier\text_classifier.py "not bad"
```

Output:

```text
Prediction for your input:
not bad -> Feedback
```

## Current Result

With the current dataset split, the script reports accuracy on the held-out test set when it runs. Because the dataset is still small, the score may vary and the model is best treated as a learning/demo project rather than a production-ready classifier.
