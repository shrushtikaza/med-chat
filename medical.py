from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

class MedicalNLP:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.ner_pipeline = pipeline("ner", 
                                   model="d4data/biomedical-ner-all",
                                   tokenizer="d4data/biomedical-ner-all",
                                   aggregation_strategy="simple")
        
        self.classifier = pipeline("text-classification", 
                                 model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
    def extract_symptoms(self, text):
        entities = self.ner_pipeline(text)
        symptoms = []
        
        for entity in entities:
            if entity['entity_group'] in ['DISEASE', 'SYMPTOM', 'BODY_PART']:
                symptoms.append(entity['word'])
        
        symptom_keywords = [
            'headache', 'fever', 'cough', 'nausea', 'vomiting', 'diarrhea',
            'chest pain', 'shortness of breath', 'fatigue', 'dizziness',
            'abdominal pain', 'back pain', 'joint pain', 'rash', 'swelling'
        ]
        
        for keyword in symptom_keywords:
            if keyword in text:
                symptoms.append(keyword)
        
        return list(set(symptoms))