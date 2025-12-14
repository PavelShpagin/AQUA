#!/usr/bin/env python3
"""
Language Detector Benchmark
Tests various SOTA language detectors on multilingual GEC data
"""

import csv
import time
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Language mapping for different detectors
LANG_MAP = {
    'en': ['en', 'eng', 'english'],
    'de': ['de', 'deu', 'ger', 'german'],
    'ua': ['uk', 'ukr', 'ukrainian', 'ua']  # Ukrainian typically uses 'uk' ISO code
}

def normalize_lang_code(detected: str, target: str) -> bool:
    """Normalize language codes across different detector formats"""
    if not detected:
        return False
    detected = detected.lower()
    return detected in LANG_MAP.get(target, [target])

def load_data(filepath: str) -> List[str]:
    """Load target texts from CSV file"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['tgt'])
    return texts

def test_langdetect(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test langdetect (Google's language-detection port)"""
    try:
        from langdetect import detect
        start = time.time()
        correct = 0
        errors = 0
        
        for text in texts:
            try:
                detected = detect(text)
                if normalize_lang_code(detected, true_lang):
                    correct += 1
            except:
                errors += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_langid(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test langid.py (py3langid)"""
    try:
        import langid
        langid.set_languages(['en', 'de', 'uk'])  # Constrain to our languages
        
        start = time.time()
        correct = 0
        
        for text in texts:
            detected, _ = langid.classify(text)
            if normalize_lang_code(detected, true_lang):
                correct += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_pycld3(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test pycld3 (Google's Compact Language Detector v3)"""
    try:
        import cld3
        start = time.time()
        correct = 0
        
        for text in texts:
            result = cld3.get_language(text)
            if result and normalize_lang_code(result.language, true_lang):
                correct += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_lingua(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test lingua-py (accurate but slower)"""
    try:
        from lingua import Language, LanguageDetectorBuilder
        
        # Map our language codes to Lingua languages
        lang_map = {
            'en': Language.ENGLISH,
            'de': Language.GERMAN,
            'ua': Language.UKRAINIAN
        }
        
        # Build detector with only our languages for speed
        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.GERMAN, Language.UKRAINIAN
        ).build()
        
        start = time.time()
        correct = 0
        
        for text in texts:
            result = detector.detect_language_of(text)
            if result:
                # Convert enum to string name (e.g., IsoCode639_1.EN -> 'en')
                detected = result.iso_code_639_1.name.lower()
                # Ukrainian in lingua uses 'uk' code
                if true_lang == 'ua' and detected == 'uk':
                    correct += 1
                elif detected == true_lang:
                    correct += 1
                    
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_fasttext(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test FastText language identification"""
    try:
        import fasttext
        import os
        
        # Download model if not exists
        model_path = 'lid.176.bin'
        if not os.path.exists(model_path):
            print("Downloading FastText language identification model...")
            import urllib.request
            url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
            urllib.request.urlretrieve(url, model_path)
        
        model = fasttext.load_model(model_path)
        
        start = time.time()
        correct = 0
        
        for text in texts:
            # FastText requires single-line text
            text = text.replace('\n', ' ')
            predictions = model.predict(text, k=1)
            if predictions[0]:
                # FastText returns '__label__en' format
                detected = predictions[0][0].replace('__label__', '')
                if normalize_lang_code(detected, true_lang):
                    correct += 1
                    
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_pycld2(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test pycld2 (Compact Language Detector 2)"""
    try:
        import pycld2 as cld2
        start = time.time()
        correct = 0
        errors = 0
        
        for text in texts:
            try:
                _, _, details = cld2.detect(text)
                if details and details[0]:
                    detected = details[0][1]  # Language code
                    if normalize_lang_code(detected, true_lang):
                        correct += 1
            except:
                errors += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_polyglot(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test Polyglot language detection"""
    try:
        from polyglot.detect import Detector
        start = time.time()
        correct = 0
        errors = 0
        
        for text in texts:
            try:
                detector = Detector(text)
                detected = detector.language.code
                if normalize_lang_code(detected, true_lang):
                    correct += 1
            except:
                errors += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_whatlang(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test whatlang-py (Rust-based, very fast)"""
    try:
        import whatlang
        start = time.time()
        correct = 0
        
        for text in texts:
            try:
                result = whatlang.detect_lang(text)
                if result:
                    # whatlang returns Lang enum, convert to string
                    detected = str(result).lower().replace('lang.', '')
                    if normalize_lang_code(detected, true_lang):
                        correct += 1
            except ValueError:
                # Language could not be detected
                pass
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_spacy_langdetect(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test spaCy with language detection"""
    try:
        import spacy
        from spacy_langdetect import LanguageDetector
        from spacy.language import Language
        
        @Language.factory("language_detector")
        def get_lang_detector(nlp, name):
            return LanguageDetector()
        
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")  # Add sentencizer for sentence boundaries
        nlp.add_pipe("language_detector", last=True)
        
        start = time.time()
        correct = 0
        
        for text in texts:
            doc = nlp(text)
            detected = doc._.language['language']
            if normalize_lang_code(detected, true_lang):
                correct += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def test_gcld3(texts: List[str], true_lang: str) -> Tuple[float, float]:
    """Test gcld3 (Google's Compact Language Detector v3 - Python binding)"""
    try:
        import gcld3
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        
        start = time.time()
        correct = 0
        
        for text in texts:
            result = detector.FindLanguage(text)
            if result.language and normalize_lang_code(result.language, true_lang):
                correct += 1
                
        elapsed = time.time() - start
        accuracy = correct / len(texts) if texts else 0
        return accuracy, elapsed
    except ImportError:
        return None, None

def main():
    # Load data
    print("Loading data...")
    data = {
        'en': load_data('data/eval/tnfn_en.csv'),
        'de': load_data('data/eval/tnfn_de.csv'),
        'ua': load_data('data/eval/tnfn_ua.csv')
    }
    
    print(f"Loaded samples: EN={len(data['en'])}, DE={len(data['de'])}, UA={len(data['ua'])}")
    print("\n" + "="*80)
    
    # Define detectors to test
    detectors = [
        ('langdetect', test_langdetect),
        ('langid.py', test_langid),
        ('pycld3', test_pycld3),
        ('gcld3', test_gcld3),
        ('lingua-py', test_lingua),
        ('fasttext', test_fasttext),
        ('pycld2', test_pycld2),
        ('polyglot', test_polyglot),
        ('whatlang-py', test_whatlang),
        ('spacy-langdetect', test_spacy_langdetect)
    ]
    
    # Store results
    results = defaultdict(dict)
    
    # Test each detector on each language
    for detector_name, detector_func in detectors:
        print(f"\nTesting {detector_name}...")
        
        total_correct = 0
        total_samples = 0
        total_time = 0
        
        for lang, texts in data.items():
            accuracy, elapsed = detector_func(texts, lang)
            
            if accuracy is not None:
                results[detector_name][lang] = {
                    'accuracy': accuracy,
                    'time': elapsed,
                    'samples': len(texts)
                }
                total_correct += accuracy * len(texts)
                total_samples += len(texts)
                total_time += elapsed
                print(f"  {lang.upper()}: {accuracy:.3f} accuracy ({elapsed:.2f}s)")
            else:
                print(f"  {detector_name} not installed - skipping")
                break
        
        if total_samples > 0:
            overall_accuracy = total_correct / total_samples
            results[detector_name]['overall'] = {
                'accuracy': overall_accuracy,
                'time': total_time,
                'samples': total_samples
            }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    # Sort by overall accuracy
    sorted_results = sorted(
        [(name, res) for name, res in results.items() if 'overall' in res],
        key=lambda x: x[1]['overall']['accuracy'],
        reverse=True
    )
    
    print("\nRanking by Overall Accuracy:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Detector':<20} {'Overall':<10} {'EN':<10} {'DE':<10} {'UA':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for rank, (name, res) in enumerate(sorted_results, 1):
        overall = res['overall']['accuracy']
        en_acc = res.get('en', {}).get('accuracy', 0)
        de_acc = res.get('de', {}).get('accuracy', 0)
        ua_acc = res.get('ua', {}).get('accuracy', 0)
        total_time = res['overall']['time']
        
        print(f"{rank:<5} {name:<20} {overall:<10.3f} {en_acc:<10.3f} {de_acc:<10.3f} {ua_acc:<10.3f} {total_time:<10.2f}")
    
    print("\n" + "="*80)
    print("Speed Analysis (samples/second):")
    print("-" * 60)
    
    speed_results = []
    for name, res in results.items():
        if 'overall' in res:
            samples_per_sec = res['overall']['samples'] / res['overall']['time']
            speed_results.append((name, samples_per_sec, res['overall']['accuracy']))
    
    speed_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Detector':<20} {'Samples/sec':<15} {'Accuracy':<10}")
    print("-" * 60)
    for name, speed, acc in speed_results:
        print(f"{name:<20} {speed:<15.1f} {acc:<10.3f}")
    
    # Installation instructions
    print("\n" + "="*80)
    print("INSTALLATION INSTRUCTIONS")
    print("="*80)
    print("""
To install missing detectors, run:

pip install langdetect        # Google's language-detection port
pip install langid            # langid.py
pip install pycld3            # Google CLD3
pip install gcld3             # Alternative CLD3 binding
pip install lingua-language-detector  # Lingua (accurate but slower)
pip install fasttext          # Facebook's FastText
pip install pycld2            # Compact Language Detector 2
pip install polyglot          # Polyglot NLP
pip install whatlang-pyo3     # Rust-based whatlang
pip install spacy spacy-langdetect  # spaCy with language detection

Note: Some packages may require additional system dependencies.
""")

if __name__ == "__main__":
    main()
