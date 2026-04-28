"""
Standalone text classification example for user messages.

The model groups messages into three categories:
- Complaint
- Feedback
- Inquiry
"""

from __future__ import annotations

import csv
import re
import string
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass(frozen=True)
class LabeledMessage:
    text: str
    label: str


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "messages.csv"

COMMON_TYPO_REPLACEMENTS = {
    "belive": "believe",
    "dot": "dont",
    "llocation": "location",
    "thsi": "this",
    "ver": "very",
}


def load_dataset(path: Path) -> List[LabeledMessage]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            LabeledMessage(text=row["text"].strip(), label=row["label"].strip())
            for row in reader
            if row.get("text") and row.get("label")
        ]


ALL_DATA = load_dataset(DATASET_PATH)

COMPLAINT_PHRASES = {
    "bad",
    "bad job",
    "charged twice",
    "crashing",
    "damaged",
    "disappointed",
    "do not use",
    "dont use",
    "frustrating",
    "ignore this",
    "issue",
    "late",
    "missing",
    "not fixed",
    "not working",
    "poor",
    "refund",
    "terrible",
    "unhappy",
    "upset",
    "awful",
}

POSITIVE_NEGATIONS = {
    "not bad",
    "not too bad",
    "not so bad",
}

FEEDBACK_PHRASES = {
    "good",
    "great",
    "nice",
    "excellent",
    "helpful",
    "useful",
    "beautiful",
    "smooth",
    "could change",
    "could improve",
    "could be better",
}

COMPLAINT_PREFIXES = (
    "do not ",
    "dont ",
    "never ",
    "stop ",
)

INQUIRY_STARTERS = (
    "how",
    "what",
    "when",
    "where",
    "which",
    "can",
    "do",
    "is",
    "are",
    "why",
)


def split_dataset(
    examples: Sequence[LabeledMessage], test_size: float = 0.25
) -> Tuple[List[LabeledMessage], List[LabeledMessage]]:
    texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    train_examples = [
        LabeledMessage(text=text, label=label)
        for text, label in zip(train_texts, train_labels)
    ]
    test_examples = [
        LabeledMessage(text=text, label=label)
        for text, label in zip(test_texts, test_labels)
    ]
    return train_examples, test_examples


def preprocess_text(text: str) -> str:
    """Lowercase text and remove punctuation for simple normalization."""
    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\s+", " ", no_punctuation).strip()
    corrected_words = [
        COMMON_TYPO_REPLACEMENTS.get(word, word)
        for word in normalized.split()
    ]
    return " ".join(corrected_words)


class TextMessageClassifier:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.model = MultinomialNB()

    def train(self, examples: Sequence[LabeledMessage]) -> None:
        texts = [example.text for example in examples]
        labels = [example.label for example in examples]
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, labels)

    def predict(self, message: str) -> str:
        fallback_label = infer_label_from_keywords(message)
        if fallback_label is not None:
            return fallback_label

        features = self.vectorizer.transform([message])
        return str(self.model.predict(features)[0])

    def predict_many(self, messages: Iterable[str]) -> List[str]:
        features = self.vectorizer.transform(list(messages))
        return [str(label) for label in self.model.predict(features)]


def evaluate_model(
    classifier: TextMessageClassifier, test_examples: Sequence[LabeledMessage]
) -> Tuple[float, str]:
    test_texts = [example.text for example in test_examples]
    actual_labels = [example.label for example in test_examples]
    predicted_labels = classifier.predict_many(test_texts)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    report = classification_report(actual_labels, predicted_labels, digits=3, zero_division=0)
    return accuracy, report


def infer_label_from_keywords(message: str) -> str | None:
    normalized = preprocess_text(message)
    if not normalized:
        return None

    if any(phrase in normalized for phrase in POSITIVE_NEGATIONS):
        return "Feedback"

    if normalized.startswith(COMPLAINT_PREFIXES):
        return "Complaint"

    if any(phrase in normalized for phrase in COMPLAINT_PHRASES):
        return "Complaint"

    if any(phrase in normalized for phrase in FEEDBACK_PHRASES):
        return "Feedback"

    first_word = normalized.split()[0]
    if first_word in INQUIRY_STARTERS:
        return "Inquiry"

    return None


def build_default_classifier(
    examples: Sequence[LabeledMessage] | None = None,
) -> TextMessageClassifier:
    classifier = TextMessageClassifier()
    classifier.train(examples or ALL_DATA)
    return classifier


def predict_category(message: str) -> str:
    """Convenience function for predicting a single message category."""
    classifier = build_default_classifier()
    return classifier.predict(message)


def run_interactive_mode(classifier: TextMessageClassifier) -> None:
    print("\nInteractive mode")
    print("Type a message to classify it, or type 'exit' to quit.")

    while True:
        try:
            user_message = input("\nEnter a message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            print("Please enter a message.")
            continue
        if user_message.lower() == "exit":
            print("Goodbye.")
            break

        print(f"Predicted category: {classifier.predict(user_message)}")


def main() -> None:
    training_data, test_data = split_dataset(ALL_DATA)
    evaluation_classifier = build_default_classifier(training_data)
    accuracy, report = evaluate_model(evaluation_classifier, test_data)
    classifier = build_default_classifier()

    print("Text Classification System")
    print("--------------------------")
    print(f"Dataset examples: {len(ALL_DATA)}")
    print(f"Training examples: {len(training_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification report:")
    print(report)

    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
        print("\nPrediction for your input:")
        print(f"{user_message} -> {classifier.predict(user_message)}")
        return

    run_interactive_mode(classifier)


if __name__ == "__main__":
    main()
