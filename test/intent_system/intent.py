# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Sample data
# texts = ["Get the weather forecast", "What's the time?", "Book a flight"]
# labels = ["get_weather", "get_time", "book_flight"]

# # Vectorize the text data
# vectorizer = CountVectorizer()
# features = vectorizer.fit_transform(texts)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# # Train a Naive Bayes classifier
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Predict intents
# predictions = model.predict(X_test)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")

# # Test a new input string
# new_text = ["birds are cool"]

# # Vectorize the new input using the same vectorizer
# new_features = vectorizer.transform(new_text)

# # Predict the intent
# new_prediction = model.predict(new_features)

# print(f"Predicted intent: {new_prediction[0]}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

# # Load data from CSV
# df = pd.read_csv("intents.csv")

# # Features and labels
# texts = df["text"]
# labels = df["intent"]

# # Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Vectorization using TF-IDF
# vectorizer = TfidfVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # Train a better model (Logistic Regression)
# model = LogisticRegression()
# model.fit(X_train_vec, y_train)

# # Evaluate
# y_pred = model.predict(X_test_vec)
# print(classification_report(y_test, y_pred))

# import re
# import json

# # Handler Registry
# handlers = {}

# def register_handler(name):
#     """Decorator to register handler functions."""
#     def decorator(func):
#         handlers[name] = func
#         return func
#     return decorator

# def pattern_to_regex(pattern):
#     """Convert a pattern with {variables} to a compiled regex."""
#     literals = []
#     variables = []
#     prev_end = 0

#     # Split pattern into literals and variables
#     for match in re.finditer(r'\{(\w+)\}', pattern):
#         var_name = match.group(1)
#         start, end = match.start(), match.end()
#         literals.append(pattern[prev_end:start])
#         variables.append(var_name)
#         prev_end = end
#     literals.append(pattern[prev_end:])  # Add remaining literal

#     regex_parts = []
#     for i in range(len(variables)):
#         # Process literal
#         lit = literals[i]
#         escaped = re.escape(lit)
#         escaped = escaped.replace(r'\ ', r'\s+')  # Allow flexible whitespace
#         regex_parts.append(escaped)

#         # Add variable capture group
#         regex_parts.append(f'(?P<{variables[i]}>.+?)')

#     # Add last literal
#     last_lit = literals[-1]
#     escaped = re.escape(last_lit)
#     escaped = escaped.replace(r'\ ', r'\s+')
#     regex_parts.append(escaped)

#     # Compile regex with case-insensitive flag
#     return re.compile(f'^{"".join(regex_parts)}$', re.IGNORECASE)

# def load_intents(intent_file):
#     """Load intents from JSON file with pattern/handler pairs."""
#     with open(intent_file) as f:
#         configs = json.load(f)

#     intents = []
#     for config in configs:
#         try:
#             regex = pattern_to_regex(config['pattern'])
#             handler = handlers[config['handler']]
#             intents.append((regex, handler))
#         except KeyError:
#             raise ValueError(f"Invalid handler: {config['handler']}")
#     return intents

# def process_input(text, intents):
#     """Process input text against registered intents."""
#     text = text.strip()
#     for regex, handler in intents:
#         if match := regex.match(text):
#             return handler(**match.groupdict())
#     return None

# # Example Usage
# @register_handler('turn_on')
# def handle_turn_on(device):
#     return {'action': 'turn_on', 'device': device}

# @register_handler('set_temperature')
# def handle_set_temp(device, temperature):
#     return {'action': 'set_temp', 'device': device, 'temp': temperature}

# if __name__ == '__main__':
#     # Load intents from file (example structure shown below)
#     intents = load_intents('intents.json')

#     # Test cases
#     print(process_input("Turn living room lights on", intents))
#     print(process_input("Set bedroom thermostat to 72 degrees", intents))
#     print(process_input("This should not match anything", intents))

# 2nd intent test

# import re
# import json
# from collections import defaultdict
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class FlexibleIntentSystem:
#     def __init__(self):
#         self.intents = []
#         self.handlers = {}
#         self.tfidf = TfidfVectorizer(stop_words='english')
#         self.pattern_vectors = {}
#         self.keyword_map = defaultdict(list)

#     def register_handler(self, name):
#         def decorator(func):
#             self.handlers[name] = func
#             return func
#         return decorator

#     def _pattern_to_regex(self, pattern):
#         pattern = re.sub(r'\s+', ' ', pattern).strip()
#         parts = re.split(r'(\{\w+\})', pattern)
#         regex_parts = []
#         variables = []

#         for part in parts:
#             if not part:
#                 continue
#             if part.startswith('{'):
#                 var_name = part[1:-1]
#                 variables.append(var_name)
#                 regex_parts.append(r'(?P<{}>.+?)'.format(var_name))
#             else:
#                 words = part.strip().split()
#                 regex_parts.append(r'(?:\w+\W+){0,3}'.join(words))

#         return re.compile(r'^{}$'.format(r'\W+'.join(regex_parts)), re.IGNORECASE), variables

#     def _preprocess_pattern(self, pattern):
#         return re.sub(r'\{\w+\}', '', pattern).strip()

#     def load_intents(self, intent_file):
#         with open(intent_file) as f:
#             configs = json.load(f)

#         patterns = []
#         for config in configs:
#             regex, variables = self._pattern_to_regex(config['pattern'])
#             clean_pattern = self._preprocess_pattern(config['pattern'])

#             self.intents.append({
#                 'regex': regex,
#                 'handler': self.handlers[config['handler']],
#                 'variables': variables,
#                 'pattern': clean_pattern
#             })
#             patterns.append(clean_pattern)
#             for word in clean_pattern.split():
#                 self.keyword_map[word.lower()].append(len(self.intents)-1)

#         # Train TF-IDF vectors for similarity matching
#         if patterns:
#             self.tfidf.fit(patterns)
#             self.pattern_vectors = self.tfidf.transform(patterns)

#     def _get_most_similar(self, text):
#         text_vec = self.tfidf.transform([text])
#         similarities = cosine_similarity(text_vec, self.pattern_vectors)
#         best_match = np.argmax(similarities)
#         return best_match if similarities[0, best_match] > 0.5 else -1

#     def process_input(self, text):
#         text = re.sub(r'\s+', ' ', text).strip().lower()

#         # 1. Try exact regex match first
#         for intent in self.intents:
#             if match := intent['regex'].match(text):
#                 return intent['handler'](**match.groupdict())

#         # 2. Fallback to keyword matching
#         keywords = set(text.split())
#         candidate_indices = set()
#         for word in keywords:
#             candidate_indices.update(self.keyword_map.get(word, []))

#         # 3. Try relaxed regex match on candidates
#         for idx in candidate_indices:
#             intent = self.intents[idx]
#             relaxed_regex = re.sub(r'\{\w+\}', '.*', intent['pattern'])
#             if re.search(relaxed_regex, text, re.IGNORECASE):
#                 # Try to extract variables using original regex
#                 if match := intent['regex'].match(text):
#                     return intent['handler'](**match.groupdict())

#         # 4. Fallback to semantic similarity
#         best_match = self._get_most_similar(text)
#         if best_match != -1:
#             intent = self.intents[best_match]
#             if match := intent['regex'].search(text):
#                 return intent['handler'](**match.groupdict())

#         return None

# # Usage Example
# system = FlexibleIntentSystem()

# @system.register_handler('turn_on')
# def handle_turn_on(device):
#     return {'action': 'turn_on', 'device': device}

# @system.register_handler('set_temp')
# def handle_set_temp(device, temperature):
#     return {'action': 'set_temp', 'device': device, 'temp': temperature}

# system.load_intents('intents.json')

# # Test cases
# print(system.process_input("turn on lights"))  # Similarity match
# print(system.process_input("please turn on the bedroom lamp")) # Relaxed match
# print(system.process_input("set temperature to 72 in kitchen")) # Out-of-order match

import re
import json
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

class LightweightIntentSystem:
    def __init__(self):
        self.intents = []
        self.handlers = {}
        self.keyword_index = defaultdict(list)

    def register_handler(self, name):
        def decorator(func):
            self.handlers[name] = func
            return func
        return decorator

    def _preprocess_text(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(token) for token in tokens]

    def _similarity(self, a, b):
        """Basic string similarity metric"""
        return SequenceMatcher(None, a, b).ratio()

    def load_intents(self, intent_file):
        with open(intent_file) as f:
            configs = json.load(f)["intents"]  # Changed to access "intents" key

        for config in configs:
            handler_name = config["handler"]  # Get handler name as string
            if handler_name not in self.handlers:
                raise ValueError(f"Handler '{handler_name}' not registered")

            intent = {
                "handler": self.handlers[handler_name],  # Get the actual function
                "patterns": [],
                "required_terms": [t.lower() for t in config.get("required", [])],
                "slots": []
            }

            # Process each pattern
            for pattern in config["patterns"]:
                # Extract slot names
                slots = re.findall(r"{(\w+)}", pattern)
                # Create clean pattern text
                clean_pattern = re.sub(r"{\w+}", "", pattern).strip()
                tokens = self._preprocess_text(clean_pattern)

                intent["patterns"].append({
                    "original": pattern,
                    "tokens": tokens,
                    "slots": slots
                })

                # Index keywords
                for token in tokens:
                    self.keyword_index[token].append(len(self.intents))

            self.intents.append(intent)

    def _extract_slots(self, text, pattern):
        """Basic slot extraction using pattern matching"""
        text = text.lower()
        slots = {}

        # Find slot positions in original pattern
        slot_positions = []
        clean_pattern = pattern["original"]
        for match in re.finditer(r"{(\w+)}", pattern["original"]):
            slot_name = match.group(1)
            start = match.start()
            end = match.end()
            slot_positions.append((slot_name, start, end))
            clean_pattern = clean_pattern.replace(match.group(0), "")

        # Reconstruct pattern parts
        parts = []
        last_end = 0
        for name, start, end in sorted(slot_positions, key=lambda x: x[1]):
            parts.append(clean_pattern[last_end:start])
            last_end = end
        parts.append(clean_pattern[last_end:])

        # Match parts against input text
        last_pos = 0
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Find part in text
            pos = text.find(part, last_pos)
            if pos == -1:
                return None  # Pattern part not found

            # Extract slot content between parts
            if i > 0 and i-1 < len(slot_positions):
                slot_name = slot_positions[i-1][0]
                slot_content = text[last_pos:pos].strip()
                if slot_content:
                    slots[slot_name] = slot_content

            last_pos = pos + len(part)

        # Extract last slot if needed
        if slot_positions and len(slot_positions) >= len(parts)-1:
            slot_name = slot_positions[-1][0]
            slot_content = text[last_pos:].strip()
            if slot_content:
                slots[slot_name] = slot_content

        return slots

    def process_input(self, text):
        text_lower = text.lower()
        tokens = self._preprocess_text(text)

        # Stage 1: Find candidate intents via keyword matching
        candidates = set()
        for token in tokens:
            if token in self.keyword_index:
                candidates.update(self.keyword_index[token])

        # Stage 2: Score candidates
        best_score = 0
        best_match = None

        for idx in candidates:
            intent = self.intents[idx]

            # Check required terms
            if not all(term in text_lower for term in intent["required_terms"]):
                continue

            # Check each pattern
            for pattern in intent["patterns"]:
                # Basic similarity check
                pattern_text = " ".join(pattern["tokens"])
                input_text = " ".join(tokens)
                similarity = self._similarity(pattern_text, input_text)

                # Threshold check
                if similarity > 0.7:
                    # Try to extract slots
                    if slots := self._extract_slots(text, pattern):
                        current_score = similarity + (0.1 * len(slots))

                        if current_score > best_score:
                            best_score = current_score
                            best_match = (intent["handler"], slots)

        if best_match and best_score > 0.75:
            handler, slots = best_match
            return handler(**slots)

        return None

# Example Usage
system = LightweightIntentSystem()

@system.register_handler('turn_on')
def handle_turn_on(device):
    return {'action': 'turn_on', 'device': device}

@system.register_handler('set_temp')
def handle_set_temp(device, temperature):
    return {'action': 'set_temp', 'device': device, 'temp': temperature}

# Create intents.json file
intents_data = {
    "intents": [
        {
            "handler": "turn_on",
            "patterns": [
                "turn on {device}",
                "activate {device}",
                "power up the {device}"
            ],
            "required": ["turn", "on"]
        },
        {
            "handler": "set_temp",
            "patterns": [
                "set {device} to {temperature}",
                "adjust {device} temperature to {temperature}",
                "make {device} {temperature}"
            ],
            "required": ["set", "temperature"]
        }
    ]
}

with open("intents.json", "w") as f:
    json.dump(intents_data, f, indent=2)

# Load and test
system.load_intents("intents.json")

# Test cases
print(system.process_input("activate the living room lights"))  # turn_on
print(system.process_input("please power up the bedroom lamp")) # turn_on
print(system.process_input("set kitchen temperature to 72"))   # set_temp
print(system.process_input("make thermostat 22"))             # set_temp
