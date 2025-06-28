import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import random

class HospitalMatcher:
    def __init__(self):
        self.hospital_data = self.create_hospital_database()
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.specialty_vectors = self.vectorizer.fit_transform(
            self.hospital_data['specialties_text']
        )
        
    def create_hospital_database(self):
        hospitals = [
            {
                'name': 'All India Institute of Medical Sciences (AIIMS)',
                'location': 'New Delhi',
                'coordinates': (28.6139, 77.2090),
                'specialties': ['cardiology', 'neurology', 'oncology', 'orthopedics', 'gastroenterology', 'nephrology'],
                'rating': 4.8,
                'emergency': True,
                'insurance': ['Government', 'CGHS', 'ECHS', 'ESI', 'Private'],
                'phone': '011-26588500',
                'type': 'Government'
            },
            {
                'name': 'Apollo Hospitals',
                'location': 'Chennai',
                'coordinates': (13.0827, 80.2707),
                'specialties': ['cardiology', 'oncology', 'neurology', 'orthopedics', 'transplant', 'emergency medicine'],
                'rating': 4.6,
                'emergency': True,
                'insurance': ['Star Health', 'HDFC ERGO', 'ICICI Lombard', 'Bajaj Allianz', 'New India Assurance'],
                'phone': '044-28293333',
                'type': 'Private'
            },
            {
                'name': 'Fortis Healthcare',
                'location': 'Gurgaon',
                'coordinates': (28.4595, 77.0266),
                'specialties': ['cardiology', 'neurology', 'oncology', 'orthopedics', 'gastroenterology', 'urology'],
                'rating': 4.4,
                'emergency': True,
                'insurance': ['Star Health', 'Max Bupa', 'HDFC ERGO', 'Religare', 'Apollo Munich'],
                'phone': '0124-4962200',
                'type': 'Private'
            },
            {
                'name': 'Medanta - The Medicity',
                'location': 'Gurgaon',
                'coordinates': (28.4089, 77.0416),
                'specialties': ['cardiology', 'neurology', 'oncology', 'transplant', 'pediatrics', 'emergency medicine'],
                'rating': 4.5,
                'emergency': True,
                'insurance': ['Star Health', 'Care Health', 'HDFC ERGO', 'ICICI Lombard', 'United India'],
                'phone': '0124-4141414',
                'type': 'Private'
            },
            {
                'name': 'Manipal Hospitals',
                'location': 'Bangalore',
                'coordinates': (12.9716, 77.5946),
                'specialties': ['cardiology', 'neurology', 'oncology', 'orthopedics', 'gastroenterology', 'nephrology'],
                'rating': 4.3,
                'emergency': True,
                'insurance': ['Star Health', 'Bajaj Allianz', 'HDFC ERGO', 'Max Bupa', 'Religare'],
                'phone': '080-25023200',
                'type': 'Private'
            },
            {
                'name': 'Max Super Speciality Hospital',
                'location': 'Delhi',
                'coordinates': (28.5355, 77.2910),
                'specialties': ['cardiology', 'neurology', 'oncology', 'orthopedics', 'emergency medicine', 'dermatology'],
                'rating': 4.2,
                'emergency': True,
                'insurance': ['Max Bupa', 'Star Health', 'HDFC ERGO', 'Care Health', 'ICICI Lombard'],
                'phone': '011-26692251',
                'type': 'Private'
            },
            {
                'name': 'Tata Memorial Hospital',
                'location': 'Mumbai',
                'coordinates': (19.0760, 72.8777),
                'specialties': ['oncology', 'radiation oncology', 'surgical oncology', 'medical oncology', 'palliative care'],
                'rating': 4.7,
                'emergency': True,
                'insurance': ['Government', 'CGHS', 'ESI', 'Star Health', 'HDFC ERGO'],
                'phone': '022-24177000',
                'type': 'Government'
            },
            {
                'name': 'King George Medical University',
                'location': 'Lucknow',
                'coordinates': (26.9124, 80.9424),
                'specialties': ['cardiology', 'neurology', 'gastroenterology', 'orthopedics', 'pediatrics', 'emergency medicine'],
                'rating': 4.1,
                'emergency': True,
                'insurance': ['Government', 'CGHS', 'ESI', 'UP State Insurance'],
                'phone': '0522-2258401',
                'type': 'Government'
            },
            {
                'name': 'Christian Medical College (CMC)',
                'location': 'Vellore',
                'coordinates': (12.9165, 79.1325),
                'specialties': ['cardiology', 'neurology', 'oncology', 'nephrology', 'gastroenterology', 'transplant'],
                'rating': 4.6,
                'emergency': True,
                'insurance': ['Government', 'Star Health', 'HDFC ERGO', 'New India Assurance'],
                'phone': '0416-2282020',
                'type': 'Private'
            },
            {
                'name': 'Narayana Health',
                'location': 'Bangalore',
                'coordinates': (12.9079, 77.6101),
                'specialties': ['cardiology', 'cardiac surgery', 'neurology', 'oncology', 'orthopedics', 'pediatrics'],
                'rating': 4.4,
                'emergency': True,
                'insurance': ['Star Health', 'HDFC ERGO', 'Bajaj Allianz', 'Care Health', 'ICICI Lombard'],
                'phone': '080-71222222',
                'type': 'Private'
            },
            {
                'name': 'PGIMER',
                'location': 'Chandigarh',
                'coordinates': (30.7333, 76.7794),
                'specialties': ['cardiology', 'neurology', 'gastroenterology', 'nephrology', 'orthopedics', 'emergency medicine'],
                'rating': 4.5,
                'emergency': True,
                'insurance': ['Government', 'CGHS', 'ESI', 'Punjab State Insurance'],
                'phone': '0172-2755555',
                'type': 'Government'
            },
            {
                'name': 'Kokilaben Dhirubhai Ambani Hospital',
                'location': 'Mumbai',
                'coordinates': (19.1136, 72.8697),
                'specialties': ['cardiology', 'neurology', 'oncology', 'transplant', 'robotic surgery', 'emergency medicine'],
                'rating': 4.3,
                'emergency': True,
                'insurance': ['Star Health', 'HDFC ERGO', 'ICICI Lombard', 'Bajaj Allianz', 'Care Health'],
                'phone': '022-30999999',
                'type': 'Private'
            }
        ]
        
        df = pd.DataFrame(hospitals)
        df['specialties_text'] = df['specialties'].apply(lambda x: ' '.join(x))
        df['insurance_text'] = df['insurance'].apply(lambda x: ' '.join(x))
        
        return df
    
    def find_hospitals_by_cashless(self, insurance_provider):
        cashless_hospitals = self.hospital_data[
            self.hospital_data['insurance'].apply(
                lambda x: insurance_provider.lower() in [ins.lower() for ins in x]
            )
        ].copy()
        
        return cashless_hospitals.sort_values('rating', ascending=False)
    
    def find_government_hospitals(self, user_location=None, max_distance=100):
        govt_hospitals = self.hospital_data[self.hospital_data['type'] == 'Government'].copy()
        
        if user_location:
            def calculate_distance(row):
                return geodesic(user_location, row['coordinates']).kilometers
            
            govt_hospitals['distance_km'] = govt_hospitals.apply(calculate_distance, axis=1)
            govt_hospitals = govt_hospitals[govt_hospitals['distance_km'] <= max_distance]
            govt_hospitals = govt_hospitals.sort_values('distance_km')
        
        return govt_hospitals
    
    def find_hospitals_by_specialty(self, required_specialty, user_location=None, max_distance=50):
        query_vector = self.vectorizer.transform([required_specialty])
        
        similarities = cosine_similarity(query_vector, self.specialty_vectors)[0]
        
        df_with_scores = self.hospital_data.copy()
        df_with_scores['specialty_match_score'] = similarities
        
        filtered_hospitals = df_with_scores[df_with_scores['specialty_match_score'] > 0.1]
        
        if user_location:
            def calculate_distance(row):
                return geodesic(user_location, row['coordinates']).kilometers
            
            filtered_hospitals = filtered_hospitals.copy()
            filtered_hospitals['distance_km'] = filtered_hospitals.apply(calculate_distance, axis=1)
            
            filtered_hospitals = filtered_hospitals[filtered_hospitals['distance_km'] <= max_distance]
            
            # Sort by combined score (specialty match + distance factor)
            filtered_hospitals['combined_score'] = (
                filtered_hospitals['specialty_match_score'] * 0.7 + 
                (1 - filtered_hospitals['distance_km'] / max_distance) * 0.3
            )
            filtered_hospitals = filtered_hospitals.sort_values('combined_score', ascending=False)
        else:
            # Sort by specialty match score only
            filtered_hospitals = filtered_hospitals.sort_values('specialty_match_score', ascending=False)
        
        return filtered_hospitals
    
    def find_hospitals_by_insurance(self, insurance_provider, user_location=None):
        matching_hospitals = self.hospital_data[
            self.hospital_data['insurance'].apply(
                lambda x: insurance_provider.lower() in [ins.lower() for ins in x]
            )
        ].copy()
        
        if user_location:
            def calculate_distance(row):
                return geodesic(user_location, row['coordinates']).kilometers
            
            matching_hospitals['distance_km'] = matching_hospitals.apply(calculate_distance, axis=1)
            matching_hospitals = matching_hospitals.sort_values('distance_km')
        
        return matching_hospitals
    
    def find_emergency_hospitals(self, user_location, max_distance=50):
        """Find nearby emergency hospitals"""
        emergency_hospitals = self.hospital_data[self.hospital_data['emergency'] == True].copy()
        
        if user_location:
            def calculate_distance(row):
                return geodesic(user_location, row['coordinates']).kilometers
            
            emergency_hospitals['distance_km'] = emergency_hospitals.apply(calculate_distance, axis=1)
            emergency_hospitals = emergency_hospitals[emergency_hospitals['distance_km'] <= max_distance]
            emergency_hospitals = emergency_hospitals.sort_values('distance_km')
        
        return emergency_hospitals
    
    def get_comprehensive_recommendation(self, specialty, user_location=None, insurance=None, emergency=False):
        if emergency:
            hospitals = self.find_emergency_hospitals(user_location)
        else:
            hospitals = self.find_hospitals_by_specialty(specialty, user_location)
        
        if insurance:
            hospitals = hospitals[
                hospitals['insurance'].apply(
                    lambda x: insurance.lower() in [ins.lower() for ins in x]
                )
            ]
        
        return hospitals.head(5)  # Return top 5 matches

class HospitalVoiceInterface:
    def __init__(self, hospital_matcher, voice_processor):
        self.hospital_matcher = hospital_matcher
        self.voice_processor = voice_processor
    
    def provide_hospital_guidance(self, hospitals_df):
        if hospitals_df.empty:
            response = "I'm sorry, I couldn't find any hospitals matching your criteria. Please try adjusting your search parameters."
        else:
            hospital_count = min(3, len(hospitals_df))  # Limit to top 3 for voice
            response = f"I found {hospital_count} hospitals for you. Here are the top recommendations: "
            
            for idx, (_, hospital) in enumerate(hospitals_df.head(3).iterrows()):
                distance_info = ""
                if 'distance_km' in hospital:
                    distance_info = f", located {hospital['distance_km']:.1f} kilometers away"
                
                response += f"{idx + 1}. {hospital['name']} in {hospital['location']}{distance_info}. "
                response += f"Rating: {hospital['rating']} out of 5. "
                response += f"Phone: {hospital['phone']}. "
                
                if hospital['type'] == 'Government':
                    response += "This is a government hospital with subsidized treatment. "
                else:
                    response += f"Accepts insurance: {', '.join(hospital['insurance'][:3])}. "
        
        return response
    
    def interactive_hospital_search(self):
        self.voice_processor.speak_response("Welcome to the Hospital Finder. I'll help you find the right hospital.")
        
        # Get specialty requirement
        self.voice_processor.speak_response("What medical specialty or condition do you need help with?")
        specialty_input = self.voice_processor.listen_to_speech()
        
        # Get location (optional)
        self.voice_processor.speak_response("What is your current city? Say 'skip' if you prefer not to specify.")
        location_input = self.voice_processor.listen_to_speech()
        user_location = None
        
        if "skip" not in location_input.lower():
            city_coordinates = {
                'delhi': (28.6139, 77.2090),
                'mumbai': (19.0760, 72.8777),
                'bangalore': (12.9716, 77.5946),
                'chennai': (13.0827, 80.2707),
                'kolkata': (22.5726, 88.3639),
                'hyderabad': (17.3850, 78.4867),
                'pune': (18.5204, 73.8567),
                'ahmedabad': (23.0225, 72.5714),
                'jaipur': (26.9124, 75.7873),
                'lucknow': (26.8467, 80.9462),
                'chandigarh': (30.7333, 76.7794),
                'gurgaon': (28.4595, 77.0266),
                'noida': (28.5355, 77.3910)
            }
            
            for city, coords in city_coordinates.items():
                if city in location_input.lower():
                    user_location = coords
                    break
        
        self.voice_processor.speak_response("Do you have health insurance? Say the name like Star Health, HDFC ERGO, or say 'government' for government schemes, or 'skip' if not applicable.")
        insurance_input = self.voice_processor.listen_to_speech()
        insurance = None if "skip" in insurance_input.lower() else insurance_input
        
        self.voice_processor.speak_response("Is this an emergency? Say yes or no.")
        emergency_input = self.voice_processor.listen_to_speech()
        is_emergency = "yes" in emergency_input.lower()
        
        hospitals = self.hospital_matcher.get_comprehensive_recommendation(
            specialty_input, user_location, insurance, is_emergency
        )
        
        guidance = self.provide_hospital_guidance(hospitals)
        self.voice_processor.speak_response(guidance)
        
        return hospitals

if __name__ == "__main__":
    hospital_matcher = HospitalMatcher()
    