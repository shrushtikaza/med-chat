import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SpecialistRecommender:
    def __init__(self):
        self.specialist_data = self.create_specialist_database()
        self.vectorizer = TfidfVectorizer()
        self.symptom_vectors = self.vectorizer.fit_transform(
            self.specialist_data['symptoms_text']
        )
    
    def create_specialist_database(self):
        specialists = {
            'Cardiologist': [
                'chest pain', 'heart palpitations', 'shortness of breath',
                'high blood pressure', 'irregular heartbeat', 'fatigue',
                'swelling in legs', 'dizziness'
            ],
            'Dermatologist': [
                'rash', 'skin irritation', 'acne', 'moles', 'hair loss',
                'nail problems', 'eczema', 'psoriasis', 'skin cancer'
            ],
            'Gastroenterologist': [
                'abdominal pain', 'nausea', 'vomiting', 'diarrhea',
                'constipation', 'bloating', 'heartburn', 'stomach pain'
            ],
            'Neurologist': [
                'headache', 'migraine', 'dizziness', 'seizures', 'memory loss',
                'numbness', 'tingling', 'weakness', 'tremor'
            ],
            'Pulmonologist': [
                'cough', 'shortness of breath', 'wheezing', 'chest pain',
                'breathing difficulty', 'lung infection'
            ],
            'Orthopedist': [
                'joint pain', 'back pain', 'muscle pain', 'bone pain',
                'fracture', 'sprain', 'arthritis', 'stiffness'
            ],
            'Psychiatrist': [
                'depression', 'anxiety', 'mood swings', 'insomnia',
                'panic attacks', 'stress', 'mental health'
            ],
            'General Practitioner': [
                'fever', 'cold', 'flu', 'general checkup', 'vaccination',
                'routine care', 'minor illness'
            ]
        }
        
        data = []
        for specialist, symptoms in specialists.items():
            data.append({
                'specialist': specialist,
                'symptoms': symptoms,
                'symptoms_text': ' '.join(symptoms)
            })
        
        return pd.DataFrame(data)
    
    def recommend_specialist(self, symptoms):
        if not symptoms:
            return "General Practitioner", 0.5
        
        query_text = ' '.join(symptoms)
        query_vector = self.vectorizer.transform([query_text])
        
        similarities = cosine_similarity(query_vector, self.symptom_vectors)[0]
        
        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx]
        
        specialist = self.specialist_data.iloc[best_match_idx]['specialist']
        
        return specialist, confidence