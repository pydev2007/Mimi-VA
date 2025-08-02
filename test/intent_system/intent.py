import spacy
from spacy.matcher import Matcher
import re

class IntentMatcher:
    def __init__(self, intents):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.intent_patterns = intents

        for intent_name, phrase_patterns in self.intent_patterns.items():
            patterns = []
            for phrase in phrase_patterns:
                if isinstance(phrase[0], dict):
                    patterns.append(phrase)
                else:
                    patterns.append([{"LEMMA": word} for word in phrase])
            self.matcher.add(intent_name, patterns)

    def extract_slots(self, doc, intent):
        slots = {"intent": intent, "action": "", "object": "", "value": ""}

        # Find the actual matched span for the action
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == intent:
                slots["action"] = doc[start:end].text
                break

        # Find object (noun chunks)
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ("dobj", "pobj", "nsubj"):
                tokens = [t for t in chunk if t.dep_ != "det"]
                cleaned_object = " ".join([t.text for t in tokens])
                slots["object"] = cleaned_object
                break

        # Find numeric values
        for ent in doc.ents:
            if ent.label_ in ("CARDINAL", "QUANTITY", "ORDINAL"):
                if re.search(r"\d+", ent.text):
                    slots["value"] = ent.text
                    break

        return slots

    def detect_intent_and_slots(self, text):
        doc = self.nlp(text.lower())
        matches = self.matcher(doc)
        if not matches:
            return {"intent": "unknown"}
        
        match_id, _, _ = matches[0]
        intent = self.nlp.vocab.strings[match_id]
        return self.extract_slots(doc, intent)