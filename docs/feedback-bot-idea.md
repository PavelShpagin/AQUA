<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameEval Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        header, nav, main, section { margin-bottom: 10px; border-radius: 8px 8px 0 0;}
        footer {position: absolute;  left: 50%;  transform: translateX(-50%); padding: 10px 0;}
        h1 { color: #1C1C1C; }
        h2, h3, h4 { color: #545454; }
        nav { background-color: #f2f2f2; padding: 10px; margin-bottom: 20px; }
        nav a { margin-right: 10px; text-decoration: none; color: #02379E; }
        nav a:hover { text-decoration: underline; }
        .metrics ul, .metrics p { margin: 10px 0; }
        .metrics table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; }
        th { background-color: #f2f2f2; }
        .legend ul { margin: 20px 0; padding: 0; list-style-type: none; }
        .legend li { margin: 5px 0; }
        .section { padding: 20px; border: 1px solid #1C1C1C; border-radius: 8px; background-color: #f9f9f9; margin-bottom: 40px; }
        .section h2 { background-color: #016A5E; color: white; padding: 10px; margin: -20px -20px 10px -20px; border-radius: 6px 6px 0 0; }
        .ner-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 100px; }
        .metrics ul li { position: relative; }
        .tooltip { visibility: hidden; width: 220px; background-color: #2551DA; color: #fff; text-align: left; border-radius: 16px 16px 16px 16px; padding: 10px; position: absolute; z-index: 1; bottom: 100%; left: 5%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; }
        .tooltip::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #2E4053 transparent transparent transparent; }
        .metrics ul li:hover .tooltip { visibility: visible; opacity: 1; }
        .metrics ul li strong { cursor: pointer; color: #000000; transition: color 0.3s ease; }
        .metrics ul li strong:hover { color: #027D7D; }
    </style>
</head>
<body>
    <header>
        <h1>FrameEval Report</h1>
    </header>
    <nav>
        <a href="#ner"><b>Technical Evaluation</b></a>
        {% if semantic_coherence is not none %}
        <a href="#information-loss"><b>Information Loss</b></a>
        {% endif %}
        {% if low_risk_documents is not none %}
        <a href="#re-identification"><b>Re-identification</b></a>
        {% endif %}
    </nav>
    <main>
        <section id="ner" class="section">
            <h2>Technical Evaluation</h2>
            <i>Assesses the performance of the Named Entity Recognition (NER) capabilities of the anonymization tool:</i>
            <div class="ner-grid">
                <div class="metrics">
                    <h3>Summary</h3>
                    <p><strong>Overall Precision:</strong> {{ (metrics.Precision * 100) | round(3) }}%</p>
                    <p><strong>Overall Recall:</strong> {{ (metrics.Recall * 100) | round(3) }}%</p>
                    <p><strong>Overall F1 Score:</strong> {{ metrics.F1 | round(3) }}</p>
                    <p><strong>№ of system alerts: </strong> {{ system_alerts_number }}</p>
                    <p><strong>№ of golden alerts: </strong> {{ golden_alerts_number }}</p>
                    <p><strong>Latency/Speed (mean):</strong> {{ speed }} ms. per input from data set</p>
                </div>
                <div class="metrics">
                    <h3>Precision per Type</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Entity Type</th>
                                <th>Coverage</th>
                                <th>Precision</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for entity, item in precision_per_type.items() %}
                            <tr>
                                <td>{{ entity }}</td>
                                <td>{{ (item[1] * item[0])|int }} out of {{ item[1]|int }}</td>
                                <td>{{ (item[0] * 100) | round(3) }}%</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="metrics">
                    <h3>Recall per Type</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Entity Type</th>
                                <th>Coverage</th>
                                <th>Recall</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for entity, item in recall_per_type.items() %}
                            <tr>
                                <td>{{ entity }}</td>
                                <td>{{ (item[1] * item[0])|int }} out of {{ item[1]|int }}</td>
                                <td>{{ (item[0] * 100) | round(3) }}%</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="metrics">
                    <h3>F1 Score per Type</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Entity Type</th>
                                <th>F1 Score</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for entity, score in f1_score_per_type.items() %}
                            <tr>
                                <td>{{ entity }}</td>
                                <td>{{ score | round(3) }}</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="metrics">
                    <h3>Labels breakdown</h3>
                    <ul>
                        <li>
                            <strong>FUL-COR:</strong> {{ metrics['FUL-COR'] }}
                            <span class="tooltip">Both the span and type of the entity match exactly with the ground truth. Classified as True Positive (TP).</span>
                        </li>
                        <li>
                            <strong>FUL-WRO:</strong> {{ metrics['FUL-WRO'] }}
                            <span class="tooltip">The span matches exactly, but the type of the entity is incorrect. Classified as TP.</span>
                        </li>
                        <li>
                            <strong>SPU:</strong> {{ metrics.SPU }}
                            <span class="tooltip">An entity was detected that does not exist in the ground truth. Classified as False Positive (FP).</span>
                        </li>
                        <li>
                            <strong>PAR-COR:</strong> {{ metrics['PAR-COR'] }}
                            <span class="tooltip">The entity partially overlaps with the correct type. Classified as False Negative (FN).</span>
                        </li>
                        <li>
                            <strong>PAR-WRO:</strong> {{ metrics['PAR-WRO'] }}
                            <span class="tooltip">The entity partially overlaps but has the wrong type. Classified as FN.</span>
                        </li>
                        <li>
                            <strong>MIS:</strong> {{ metrics.MIS }}
                            <span class="tooltip">An entity present in the ground truth was not detected. Classified as FN.</span>
                        </li>
                    </ul>
                </div>
            </div>
        </section>

        {% if semantic_coherence is not none %}
        <section id="information-loss" class="section">
            <h2>Information Loss</h2>
            <i>Assess the impact of data anonymization on text by measuring perplexity and semantic coherence. Increased perplexity suggests a loss of naturalness, while cosine similarity of semantic embeddings checks for meaning retention:</i>
            <div class="ner-grid">
                <div class="metrics">
                    <h3>Summary</h3>
                    <p><strong>Information Loss Score:</strong> {{ information_score }}</p>
                    <p><strong>Information Loss Category:</strong> {{ information_loss_category }}</p>
                </div>
                <div class="metrics">
                    <h3>Perplexity</h3>
                    <p><strong>Original Perplexity:</strong> {{ perplexity_original }}</p>
                    <p><strong>Anonymized Perplexity:</strong> {{ perplexity_anonymized }}</p>
                </div>
                <div class="metrics">
                    <h3>Semantic Coherence <span class="tooltip">Measures the consistency and logical connection in the anonymized text.</span></h3>
                    <p><strong>Cosine Similarity:</strong> {{ semantic_coherence * 100 | round(2) }}%</p>
                </div>
            </div>
        </section>
        {% endif %}

        {% if low_risk_documents is not none %}
        <section id="re-identification" class="section">
            <h2>Re-identification</h2>
            <i>Assesses the effectiveness of anonymization methods in safeguarding the identities of individuals in a dataset against re-identification attempts:</i>
            <div class="ner-grid">
                <div class="metrics">
                    <h3>Summary</h3>
                    <p><strong>Identity Protection Score:</strong> {{ identity_protection_score }} (higher is better)</p>
                    <p><strong>Re-identification Risk Score:</strong> {{ re_identification_risk_score }} (lower is better)</p>
                </div>
                <div class="metrics">
                    <h3>Risky Documents</h3>
                    <p><strong>High Risk Documents:</strong> {{ high_risk_documents }} ({{ (high_risk_documents / (high_risk_documents + low_risk_documents) * 100) | round(2) }}%)</p>
                    <p><strong>Low Risk Documents:</strong> {{ low_risk_documents }} ({{ (low_risk_documents / (high_risk_documents + low_risk_documents) * 100) | round(2) }}%)</p>
                </div>
                <div class="metrics">
                    <h3>Identifiers</h3>
                    <p><strong>Direct:</strong> {{ masked_direct_mentions }} masked out of {{ total_direct_mentions }} {% if total_direct_mentions > 0 %}({{ (masked_direct_mentions / total_direct_mentions * 100) | round(2) }}%){% else %}(0%){% endif %}</p>
                    <p><strong>Quasi:</strong> {{ masked_quasi_mentions }} masked out of {{ total_quasi_mentions }} {% if total_quasi_mentions > 0 %}({{ (masked_quasi_mentions / total_quasi_mentions * 100) | round(2) }}%){% else %}(0%){% endif %}</p>
                </div>
            </div>
        </section>
        {% endif %}
    </main>
    <footer>
        <p>&copy; 2024 FrameEval Report</p>
    </footer>
</body>
</html>



import copy
import csv
import json
import os
from difflib import SequenceMatcher

from grampy.text import AnnotatedText
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from typing import List, Optional

ALLOWED_TO_SKIP = [
    "the",
    "from",
    "http://",
    "https://",
    "www.",
    "'s",
    "$",
    "<",
    "mrs",
    "mrs.",
    "mr",
    "ms.",
    "mr.",
]

# See Annotation Guidelines: https://docs.google.com/document/d/1MO6OAGlPCqU9-AGjpGEEmcP1PJH6A4bjNISFK9N13Ao/edit
NORMALIZATION_DICT = {
    'EMAIL': 'WEB_SOCIAL_EMAIL',
    'ID': 'ID_NUMBER',
    'IP': 'WEB_SOCIAL_EMAIL',
    'LOC': 'LOCATION',
    'ORG': 'ORGANIZATION',  # Also needed for SpaCy
    'PASSWORD': 'ID_NUMBER',  # Also needed for Private AI.
    'PER': 'PERSON',
    'PHONE': 'DIGIT',
    'SSN': 'ID_NUMBER',  # Also needed for Private AI.
    'URL': 'WEB_SOCIAL_EMAIL',  # Also needed for Private AI.
    'USERNAME': 'WEB_SOCIAL_EMAIL',  # Also needed for Private AI.
    'VEHICLE_PLATE': 'ID_NUMBER',
    'WEB_MAIL_SOCIAL': 'WEB_SOCIAL_EMAIL',

    # GLiNER translated entity labels mappings
    # English
    'person': 'PERSON',
    'organization': 'ORGANIZATION',
    'phone number': 'PHONE',
    'IP address': 'WEB_SOCIAL_EMAIL',
    'location': 'LOCATION',
    'ID number': 'ID_NUMBER',
    'email': 'WEB_SOCIAL_EMAIL',
    'user name': 'WEB_SOCIAL_EMAIL',
    'date': 'DATE',
    'monetary values': 'MONETARY_VALUES',

    # French
    'personne': 'PERSON',
    'organisation': 'ORGANIZATION',
    'numéro de téléphone': 'PHONE',
    'adresse IP': 'WEB_SOCIAL_EMAIL',
    'lieu': 'LOCATION',
    "numéro d'identification": 'ID_NUMBER',
    'nom d\'utilisateur': 'WEB_SOCIAL_EMAIL',
    'valeurs monétaires': 'MONETARY_VALUES',

    # German
    'telefonnummer': 'PHONE',
    'IP-Adresse': 'WEB_SOCIAL_EMAIL',
    'ort': 'LOCATION',
    'ausweisnummer': 'ID_NUMBER',
    'benutzername': 'WEB_SOCIAL_EMAIL',
    'datum': 'DATE',
    'geldbeträge': 'MONETARY_VALUES',

    # Italian
    'persona': 'PERSON',
    'organizzazione': 'ORGANIZATION',
    'numero di telefono': 'PHONE',
    'indirizzo IP': 'WEB_SOCIAL_EMAIL',
    'luogo': 'LOCATION',
    'numero identificativo': 'ID_NUMBER',
    'nome utente': 'WEB_SOCIAL_EMAIL',
    'data': 'DATE',
    'valori monetari': 'MONETARY_VALUES',

    # Portuguese
    'pessoa': 'PERSON',
    'organização': 'ORGANIZATION',
    'número de telefone': 'PHONE',
    'endereço IP': 'WEB_SOCIAL_EMAIL',
    'localização': 'LOCATION',
    'número de identificação': 'ID_NUMBER',
    'nome de usuário': 'WEB_SOCIAL_EMAIL',
    'valores monetários': 'MONETARY_VALUES',

    # Spanish
    'organización': 'ORGANIZATION',
    'número de teléfono': 'PHONE',
    'dirección IP': 'WEB_SOCIAL_EMAIL',
    'ubicación': 'LOCATION',
    'número de identificación': 'ID_NUMBER',
    'correo electrónico': 'WEB_SOCIAL_EMAIL',
    'nombre de usuario': 'WEB_SOCIAL_EMAIL',
    'fecha': 'DATE',
    'valores monetarios': 'MONETARY_VALUES',

    # Ukrainian
    'особа': 'PERSON',
    'організація': 'ORGANIZATION',
    'номер телефону': 'PHONE',
    'IP-адреса': 'WEB_SOCIAL_EMAIL',
    'місце': 'LOCATION',
    'ідентифікаційний номер': 'ID_NUMBER',
    'електронна пошта': 'WEB_SOCIAL_EMAIL',
    'ім\'я користувача': 'WEB_SOCIAL_EMAIL',
    'дата': 'DATE',
    'грошові значення': 'MONETARY_VALUES',

    # SpaCy: See https://www.kaggle.com/code/curiousprogrammer/entity-extraction-and-classification-using-spacy for
    # details
    'CARDINAL': 'DIGIT',
    'EVENT': 'O',  # Also needed for Private AI.
    'FAC': 'MISC',
    'GPE': 'LOCATION',
    'LANGUAGE': 'DEM',  # Also needed for Private AI.
    'LAW': 'O',
    'MONEY': 'MONETARY_VALUES',  # Also needed for Private AI.
    'NORP': 'DEM',
    'ORDINAL': 'DIGIT',
    'PERCENT': 'O',
    'QUANTITY': 'O',
    'TIME': 'O',  # Also needed for Private AI.
    'WORK_OF_ART': 'O',

    # Private AI: See https://docs.private-ai.com/entities/ for examples and details
    'ACCOUNT_NUMBER': 'ID_NUMBER',
    'AGE': 'DEM',
    'BANK_ACCOUNT': 'ID_NUMBER',
    'BLOOD_TYPE': 'O',
    'CONDITION': 'O',
    'CORPORATE_ACTION': 'O',
    'CREDIT_CARD': 'ID_NUMBER',
    'CREDIT_CARD_EXPIRATION': 'DATE',
    'CVV': 'ID_NUMBER',
    'DATE_INTERVAL': 'DATE',
    'DAY': 'DATE',
    'DOB': 'DATE',
    'DOSE': 'O',
    'DRIVER_LICENSE': 'ID_NUMBER',
    'DRUG': 'O',
    'DURATION': 'O',
    'EFFECT': 'O',
    'EMAIL_ADDRESS': 'WEB_SOCIAL_EMAIL',
    'FILENAME': 'O',
    'FINANCIAL_METRIC': 'O',
    'GENDER': 'O',
    'HEALTHCARE_NUMBER': 'ID_NUMBER',
    'INJURY': 'O',
    'IP_ADDRESS': 'WEB_SOCIAL_EMAIL',
    'LOCATION_ADDRESS': 'LOCATION',
    'LOCATION_ADDRESS_STREET': 'LOCATION',
    'LOCATION_CITY': 'LOCATION',
    'LOCATION_COORDINATE': 'O',
    'LOCATION_COUNTRY': 'LOCATION',
    'LOCATION_STATE': 'LOCATION',
    'LOCATION_ZIP': 'LOCATION',
    'MARITAL_STATUS': 'DEM',
    'MEDICAL_CODE': 'O',
    'MEDICAL_PROCESS': 'O',
    'MONTH': 'DATE',
    'NAME': 'PERSON',
    'NAME_FAMILY': 'PERSON',
    'NAME_GIVEN': 'PERSON',
    'NAME_MEDICAL_PROFESSIONAL': 'PERSON',
    'NUMERICAL_PII': 'ID_NUMBER',
    'OCCUPATION': 'DEM',
    'ORGANIZATION_ID': 'O',
    'ORGANIZATION_MEDICAL_FACILITY': 'ORGANIZATION',
    'ORIGIN': 'DEM',
    'PASSPORT_NUMBER': 'ID_NUMBER',
    'PHONE_NUMBER': 'PHONE',
    'PHYSICAL_ATTRIBUTE': 'DEM',
    'POLITICAL_AFFILIATION': 'DEM',
    'PROJECT': 'O',
    'RELIGION': 'DEM',
    'ROUTING_NUMBER': 'ID_NUMBER',
    'SEXUALITY': 'DEM',
    'STATISTICS': 'O',
    'TREND': 'O',
    'VEHICLE_ID': 'ID_NUMBER',
    'YEAR': 'DATE',
    'ZODIAC_SIGN': 'O'
}

# Overrides
NORMALIZATION_DICT_OVERRIDES = {
    # The various SpaCy models use LOC for points of interest, which we tag as MISC.  Countries/cities/states are
    # tagged instead as GPE (Geopolitical Entity), which is mapped to LOCATION above.
    "spacy_sm": {
        'LOC': 'MISC',
    },
    "spacy_md": {
        'LOC': 'MISC',
    },
    "spacy_lg": {
        'LOC': 'MISC',
    },
    "spacy_trf": {
        'LOC': 'MISC',
    },
    "private_ai": {
        # Private AI uses LOCATION for points of interest, which we tag as MISC.  What we call "LOCATION" Private AI
        # tags with more specific entity types, such as "LOCATION_CITY", "LOCATION_COUNTRY", etc.
        'LOCATION': 'MISC'
    }
}


def build_normalization_dict(anonymizer):
    normalization_dict = copy.deepcopy(NORMALIZATION_DICT)
    if anonymizer in NORMALIZATION_DICT_OVERRIDES:
        normalization_dict.update(NORMALIZATION_DICT_OVERRIDES.get(anonymizer))

    return normalization_dict


class TextDataset:
    def __init__(self, source):
        if isinstance(source, str):
            self.filepath = source
            self.texts = self._read_file()
        elif isinstance(source, list):
            self.filepath = None
            self.texts = source
        else:
            raise ValueError("Source must be a file path or a list of texts")

        self.original_data = []
        self.anonymized_data = []
        self._prepare_data()

    def _read_file(self):
        with open(self.filepath, "r") as file:
            return [line.strip() for line in file.readlines()]

    def _prepare_data(self):
        for sentence in tqdm(self.texts, desc="Preparing dataset"):
            anno_sentence = AnnotatedText(sentence)
            self.original_data.append(anno_sentence.get_original_text())
            self.anonymized_data.append(anno_sentence.get_corrected_text())

    def __len__(self):
        return len(self.texts)

    def get_original_data(self):
        return self.original_data

    def get_anonymized_data(self):
        return self.anonymized_data


def map_tokens_to_characters(token_spans):
    """Creates a mapping of token indices to character-level spans in the sentence."""
    token_to_char_span = {}
    for idx, (start, end) in enumerate(token_spans):
        token_to_char_span[idx] = (start, end)
    return token_to_char_span


def extend_alerts_with_char_info(token_spans, alerts):
    """Extends alerts with character-level information."""
    try:
        token_to_char_span = map_tokens_to_characters(token_spans)
        for alert in alerts:
            begin_token = alert["begin"] if alert["begin"] != -1 else 0
            end_token = alert["end"]
            begin_char = token_to_char_span[begin_token][0]
            end_char = token_to_char_span[end_token - 1][1]

            alert.update({"begin_char": begin_char, "end_char": end_char})

        return alerts
    except Exception as e:
        print(f"Failed to extend alerts with character-level information: {e}")
        print(e)


def merge_enclosed_entities(golden_entities, system_entities):
    """Merge enclosed entities in system_entities."""
    for golden in golden_entities:
        if golden["type"] == "LOCATION":
            enclosed_entities = [
                entity
                for entity in system_entities
                if golden["begin"] <= entity["begin"]
                   and entity["end"] <= golden["end"]
                   and not (
                        golden["begin"] == entity["begin"]
                        and entity["end"] == golden["end"]
                )
            ]

            if enclosed_entities and len(enclosed_entities) >= 2:
                new_begin = min([entity["begin"] for entity in enclosed_entities])
                new_end = max([entity["end"] for entity in enclosed_entities])

                merged_entity = {"begin": new_begin, "end": new_end, "type": "LOCATION"}

                system_entities = [
                    entity
                    for entity in system_entities
                    if entity not in enclosed_entities
                ]
                system_entities.append(merged_entity)

    return system_entities


def convert_alerts_to_char_based(row):
    token_spans = row["TokenSpans"]
    alerts = row["Alerts"]
    extended_alerts = extend_alerts_with_char_info(token_spans, alerts)
    return extended_alerts


def find_uncommon_part(str1, str2):
    if not str1 or not str2:
        return None

    longest = max(str1, str2, key=len)
    shortest = min(str1, str2, key=len)

    if shortest in longest:
        return max(longest.split(shortest, 1), key=len).strip()
    else:
        return None


def get_similarity(str1, str2):
    """Return a similarity ratio between two strings (0 to 1)."""
    return SequenceMatcher(None, str1, str2).ratio()


def spans_exact_match(gold_ent, system_ent, paragraph, similarity_threshold=0.9):
    """Return True if two spans are similar enough based on position and text similarity."""
    gold_start, gold_end, gold_type = (
        gold_ent["begin"],
        gold_ent["end"],
        gold_ent["type"],
    )
    system_start, system_end = system_ent["begin"], system_ent["end"]

    # If start and end positions are exactly the same, return True
    if (gold_start, gold_end) == (system_start, system_end):
        return True

    # Check if spans overlap
    elif spans_overlap(gold_ent, system_ent):
        gold_entity_text = paragraph[gold_start:gold_end]
        system_entity_text = paragraph[system_start:system_end]
        uncommon = find_uncommon_part(gold_entity_text, system_entity_text)

        # Compute similarity between texts
        similarity_score = get_similarity(gold_entity_text, system_entity_text)

        # If similarity is above the threshold, consider it a match
        if similarity_score >= similarity_threshold or (
                similarity_score >= 0.7
                and gold_type
                in ["WEB_MAIL_SOCIAL", "PHONE", "ID_NUMBER", "MONETARY_VALUES"]
        ):
            return True

        # Consider it a match if uncommon part is allowed or meets criteria
        return uncommon and (
                uncommon in ALLOWED_TO_SKIP
                or (uncommon.isdigit() and gold_type == "LOCATION")
        )
    else:
        return False


def spans_overlap(span1, span2):
    """
    Return True if two spans partially or fully overlap.
    """
    return max(span1["begin"], span2["begin"]) <= min(span1["end"], span2["end"])


def classify_span(gold_ent, system_ent, text):
    """
    Classify the relationship between a golden entity and a system entity.
    """
    if spans_exact_match(gold_ent, system_ent, text):
        return "exact"
    elif spans_overlap(gold_ent, system_ent):
        return "partial"
    else:
        return "no_match"


class HtmlReportGenerator:
    def __init__(self, template_path: str = "report_template.html"):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_dir = os.path.join(base_path, "report")

        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template_path = template_path

        try:
            self.template = self.env.get_template(os.path.basename(template_path))
        except Exception as e:
            print(f"Failed to load template: {e}")
            raise

    def render(self, data: dict) -> str:
        """
        Render the template with the given data.

        Args:
            data (dict): Data containing metrics and other information to populate the template.

        Returns:
            str: Rendered HTML output.
        """
        output = self.template.render(
            metrics=data["metrics"],
            system_alerts_number=data["metrics"]["Total System Alerts"],
            golden_alerts_number=data["metrics"]["Total Golden Alerts"],
            precision_per_type=data["metrics"]["Precision per type"],
            recall_per_type=data["metrics"]["Recall per type"],
            f1_score_per_type=data["metrics"]["F1 Score per type"],
            perplexity_original=data.get("perplexity_original", None),
            perplexity_anonymized=data.get("perplexity_anonymized", None),
            semantic_coherence=data.get("semantic_coherence", None),
            information_score=data.get("information_loss_score", None),
            information_loss_category=data.get("information_loss_category", ""),
            identity_protection_score=data.get("identity_protection_score", None),
            high_risk_documents=data.get("high_risk_documents", None),
            low_risk_documents=data.get("low_risk_documents", None),
            masked_direct_mentions=data.get("masked_direct_mentions", None),
            total_direct_mentions=data.get("total_direct_mentions", None),
            masked_quasi_mentions=data.get("masked_quasi_mentions", None),
            total_quasi_mentions=data.get("total_quasi_mentions", None),
            re_identification_risk_score=data.get("re_identification_risk_score", None),
            speed=data["speed"],
        )
        return output

    def save_to_file(self, output: str, file_path: str):
        """
        Save the rendered HTML output to a file.

        Args:
            output (str): Rendered HTML output.
            file_path (str): Path to save the output file.
        """
        with open(file_path, "w") as f:
            f.write(output)


def write_raw_results_file(output_file_prefix, anonymizer, original_sentences, golden_alerts, system_alerts,
                           processed_golden_alerts, processed_system_alerts, api_responses: Optional[List] = None):
    test_case_number = 0
    fieldnames = ["number", "text", "ref_entities_raw", "hyp_entities_raw", "ref_entities_processed",
                  "hyp_entities_processed", "only_in_ref_entities_processed", "only_in_hyp_entities_processed"]
    if api_responses is not None:
        fieldnames.append("full_api_response")

    else:
        api_responses = [None] * len(original_sentences)

    with (open(f"./frame_eval/report/{output_file_prefix}_results.csv", "w", newline="", encoding="utf-8") as
          results_file):
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        for text, ref_entities, hyp_entities, processed_ref_entities, processed_hyp_entities, api_response in zip(
            original_sentences, golden_alerts, system_alerts, processed_golden_alerts, processed_system_alerts,
            api_responses
        ):
            test_case_number += 1

            # Extract values from text using indices.
            for item in processed_ref_entities:
                if item.get("value") is None:
                    item.update({"value": text[item.get("begin"):item.get("end")]})

            for item in processed_hyp_entities:
                if item.get("value") is None:
                    item.update({"value": text[item.get("begin"):item.get("end")]})

            data_to_write = {
                "number": test_case_number,
                "text": text,
                "ref_entities_raw": json.dumps(ref_entities, ensure_ascii=False),
                "hyp_entities_raw": json.dumps(hyp_entities, ensure_ascii=False),
                "ref_entities_processed": json.dumps(processed_ref_entities, ensure_ascii=False),
                "hyp_entities_processed": json.dumps(processed_hyp_entities, ensure_ascii=False),
                "only_in_ref_entities_processed": json.dumps([item for item in processed_ref_entities if item not in
                                                              processed_hyp_entities], ensure_ascii=False),
                "only_in_hyp_entities_processed": json.dumps([item for item in processed_hyp_entities if item not in
                                                              processed_ref_entities], ensure_ascii=False)
            }
            if api_response is not None:
                data_to_write.update({"full_api_response": json.dumps(api_response, ensure_ascii=False)})

            writer.writerow(data_to_write)




cf1a121ee29dd41f7043b05fb045d239ca463235



Given the example information above, we'll need to develop a new feedback_bot that will take as input feedback data in the data/feedback folder, (just also --input specifying the csv file name) and will generate the full html (with styling) that will contain all the analysis about the FP1/FP2/FP3/TP of the edits in the data, and basically how severe or not each edit. The html report should contain:
1. report summary and overview of the most problematic model areas
2. piechart of FP1/FP2/FP3/TP distribution ;; ALso a graph of FP1/FP2/FP3 per writing_type (e.g. 3 aligned bars for each of the 4 categories with the highest error rate, see attached screenshot for the rough idea).
3. for each, starting from FP1 to FP3 there is a summary on common model error patterns from that category. Highlights of a few representative examples or citations to a few examples in the corpus. Same for TP, but instead, highlight model strenghts instead of errors.
4. Conclusion, and directions, where the model developement should focus on.
5. Reference examples section, that will have all the reference examples from the main section. Should be formatted like research paper: [1], [2], etc.

Here is the algorithm for creating FP1...FP3, TP summaries:
1. Using ./feedback_bot/run.sh Run judge with the config in the feedback_bot/config.yaml
2. The ./run.sh will call the shell/run_judge.sh --config config.yaml and label the feedback data with TP/FP3/FP2/FP1 labels, writing_types, reasoning, and writing the result to the data/feedback/processed folder
3. Then, the generator will be called that generates the final, visually appealing html with grammarly styling (inspiration html generator and html itself is above)
4. The generator will call the cluster.py* first, and then take the summaries for each category and ranked examples -- and will also run a summarizer gpt-4o model on all the data to generate the main summary and conclusion. This all is then passed to the html arguments and the final html is rendered.

*cluster.py works like this:
1. Takes the processed csv file, converts each reasoning into a vector using Sentence-bert off-the-shelf (best one for this task), then for each separate FP1/FP2/FP3/TP category, use DBSCAN or HDBSCAN to cluster each category separately, for example, FP1s will have some clusters, then FP2s -- completely independent from FP1 clusters etc. Then, for each category, and each cluster, take 5-10 most representative samples, and use those as in context examples that for the specific TP/FP3/FP2/FP1 category will use gpt-4o to generate the error pattern name and quick summary on the model error (if FP3/FP2/FP1);;; otherwise, for TP, generate model strenghts from the cluster representatives and provide brief summary. So, in the end of cluster.py, for each separate category FP1/FP2/FP3/TP generate a list of pattern names, summaries, and examples list.

The main feedback_bot/generator.py can take as argument --backend to use instead of gpt-4o for summaries etc.   Also --cluster argument with the clustering algorithm (e.g. DBSCAN, HDBSCAN, KMEANS etc.), also --embedding (bert/tfidf etc.) for the flexibility. Write the full script and implementation of this bot, and test it in the browser, making sure the report and algorithms work completely.

Note, that the report UI should be Grammarly style and super cool and polished, really appealing, no boring or generic UI, green tones, cool piechart and bar, super modern and cool, appealing, grammarly.

If you have any insights, questions or suggestion to change the algorithm above, feel free to tell me that. Overall, what do you think about the algorithm and idea? So, build or ask.





Also update the README.md with minimal instructions how to reproduce the feedback bot report generation.
User has to get raw feedback data, for example by using the fql query in the feedback (https://feedbacks.grammarly.io/fql ): 

select
  CONCAT(substr(sentence, 1, start_pos),
         '{', substr(sentence, start_pos + 1, end_pos - start_pos), '=>', replacements[1], '}',
         substr(sentence, end_pos + 1)) as alert

from examples

where
  CARDINALITY(replacements) > 0
  AND LENGTH(sentence) >= 30
  AND dt >= '2025-08-01'
  AND dt <= '2025-08-20'

limit 1000

Then, they should place the file into the data/feedback folder.

Then, use the csv name as --input argument to ./feedback_bot/run.sh and run the processing. The run.sh also should right away create the html and open browser with that html path.



