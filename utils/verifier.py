import os
import requests
import spacy
import re
from textblob import TextBlob
from typing import Tuple, Dict, List, Any
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Constants ---
FACT_CHECK_SOURCES = {
    "Politifact": "https://www.politifact.com/search/?q=",
    "Snopes": "https://www.snopes.com/?s=",
    "GoogleFactCheck": "https://toolbox.google.com/factcheck/explorer/search/",
    "FactCheck.org": "https://www.factcheck.org/?s=",
    "Full Fact": "https://fullfact.org/search/?q=",
}

SENSATIONAL_PATTERNS = [
    r"\b(urgent|breaking|shocking|exposed|secret|exclusive|revealed)\b",
    r"\b(they don't want you to know|hidden truth|mainstream media won't tell you|you won't believe)\b",
    r"(! ){3,}", r"[A-Z]{10,}",
    r"\b(you'll never guess|this is unbelievable|this will change everything)\b"
]

UNRELIABLE_DOMAINS = [
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "yournewswire.com", "worldtruth.tv", "thegatewaypundit.com"
]

CLICKBAIT_PHRASES = [
    "you won't believe", "what happened next", "doctors hate this",
    "this one trick", "the reason will shock you", "this is why",
    "find out why", "the shocking truth about"
]

# --- NLP Setup ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Load ML Model ---
try:
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    model = joblib.load(os.path.join(MODEL_PATH, 'fake_news_model.pkl'))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
except Exception as e:
    model = None
    vectorizer = None
    print(f"[ERROR] {datetime.now()}: Model loading failed: {e}")

# --- Helper Functions ---
def log_error(error: str) -> None:
    print(f"[ERROR] {datetime.now()}: {error}")

def check_sensational_language(text: str) -> bool:
    return any(re.search(pattern, text.lower()) for pattern in SENSATIONAL_PATTERNS)

def check_unreliable_source(text: str) -> bool:
    return any(domain in text.lower() for domain in UNRELIABLE_DOMAINS)

def check_clickbait(text: str) -> bool:
    return any(phrase in text.lower() for phrase in CLICKBAIT_PHRASES)

def check_grammar_quality(text: str) -> bool:
    doc = nlp(text)
    if len(list(doc.sents)) < 1:
        return False
    errors = sum(1 for sent in doc.sents if len([t for t in sent if t.pos_ in ('NOUN', 'VERB')]) < 2)
    return errors / max(1, len(list(doc.sents))) < 0.4

def sentiment_analysis(text: str) -> str:
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.3:
        return "Positive"
    elif blob.sentiment.polarity < -0.3:
        return "Negative"
    else:
        return "Neutral"

def wikidata_check(person: str, fact: str) -> Tuple[bool, str]:
    try:
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": person
        }
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        data = response.json()
        if not data.get("search"):
            return False, "Entity not found in Wikidata"

        qid = data["search"][0]["id"]
        detail_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        detail_response = requests.get(detail_url)
        detail_response.raise_for_status()
        detail_data = detail_response.json()

        if "P19" in detail_data["entities"][qid]["claims"]:
            birth_place = detail_data["entities"][qid]["claims"]["P19"][0]["mainsnak"]["datavalue"]["value"]["text"]
            if fact.lower() in birth_place.lower():
                return True, f"Birthplace confirmed as {birth_place}"
            return False, f"Birthplace is {birth_place} (claimed: {fact})"
        return False, "No relevant data found in Wikidata"
    except Exception as e:
        log_error(f"Wikidata check failed: {e}")
        return False, "Verification service unavailable"

def fact_check_claim(claim: str) -> Dict[str, str]:
    results = {}
    try:
        for source, base_url in FACT_CHECK_SOURCES.items():
            results[source] = base_url + requests.utils.quote(claim)
    except Exception as e:
        log_error(f"Fact-check search failed: {e}")
    return results

def extract_entities(text: str) -> Tuple[List[str], List[str]]:
    try:
        doc = nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
        return persons, locations
    except Exception as e:
        log_error(f"Entity extraction failed: {e}")
        return [], []

# --- Main Verification Function ---
def verify_news(text: str) -> Dict[str, Any]:
    result = {
        "text": text,
        "final_verdict": "UNVERIFIED",
        "reason": "Insufficient evidence",
        "red_flags": {},
        "entity_verification": [],
        "fact_check_links": {},
        "quality_metrics": {},
        "ml_prediction": None,
        "ml_confidence": None
    }

    # Heuristic Checks
    red_flags = {
        "sensational_language": check_sensational_language(text),
        "unreliable_source": check_unreliable_source(text),
        "clickbait_phrases": check_clickbait(text),
        "poor_grammar": not check_grammar_quality(text)
    }
    result['red_flags'] = red_flags

    if sum(red_flags.values()) >= 2:
        result.update({
            "final_verdict": "FAKE",
            "reason": "Multiple red flags detected"
        })

    # Entity Verification
    persons, locations = extract_entities(text)
    entity_results = []
    fact_check_links = {}

    if persons and locations:
        for person in persons[:3]:
            for location in locations[:3]:
                verified, reason = wikidata_check(person, location)
                entity_results.append({
                    "person": person,
                    "location": location,
                    "verified": verified,
                    "reason": reason
                })
                claim = f"{person} {location}"
                fact_check_links[claim] = fact_check_claim(claim)

    result['entity_verification'] = entity_results
    result['fact_check_links'] = fact_check_links

    verified_flags = [e['verified'] for e in entity_results]

    if entity_results:
        if all(verified_flags):
            result.update({"final_verdict": "VERIFIED", "reason": "All claims verified via Wikidata"})
        elif any(verified_flags):
            result.update({"final_verdict": "PARTIALLY_VERIFIED", "reason": "Some claims verified"})
        elif any(v is False for v in verified_flags) and result['final_verdict'] != "FAKE":
            result.update({"final_verdict": "SUSPICIOUS", "reason": "Some claims contradict known facts"})

    # ML Prediction
    if model and vectorizer:
        try:
            vect_text = vectorizer.transform([text])
            prediction = model.predict(vect_text)[0]
            confidence = model.predict_proba(vect_text).max()

            result['ml_prediction'] = prediction
            result['ml_confidence'] = round(float(confidence), 4)

            if result['final_verdict'] == "UNVERIFIED":
                result.update({
                    "final_verdict": "REAL" if prediction == "REAL" else "FAKE",
                    "reason": f"ML model suggests {prediction} with {round(confidence*100)}% confidence"
                })

        except Exception as e:
            log_error(f"ML prediction failed: {e}")
            result['ml_prediction'] = "ERROR"
            result['ml_confidence'] = 0.0

    # Suspicious fallback
    if any(red_flags.values()) and result['final_verdict'] == "UNVERIFIED":
        result.update({
            "final_verdict": "SUSPICIOUS",
            "reason": "Red flags detected but unverifiable via Wikidata"
        })

    # Quality Metrics
    sentiment = sentiment_analysis(text)
    result['quality_metrics'] = {
        "word_count": len(text.split()),
        "proper_nouns": len(persons),
        "locations": len(locations),
        "avg_sentence_length": sum(len(sent.text.split()) for sent in nlp(text).sents) / max(1, len(list(nlp(text).sents))),
        "sentiment": sentiment
    }

    return result
