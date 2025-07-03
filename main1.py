import os
import sys
from typing import Optional, Tuple, List

# Import all the custom modules
from speech_rec import VoiceProcessor
from medical import MedicalNLP
from specialist import SpecialistRecommender
from hospital import HospitalMatcher, HospitalVoiceInterface

class MedicalVoiceAssistant:
    def __init__(self):
        print("Initializing Medical Voice Assistant...")
        
        try:
            # Initialize all components
            self.voice_processor = VoiceProcessor()
            self.medical_nlp = MedicalNLP()
            self.specialist_recommender = SpecialistRecommender()
            self.hospital_matcher = HospitalMatcher()
            self.hospital_voice_interface = HospitalVoiceInterface(
                self.hospital_matcher, 
                self.voice_processor
            )
            
            # City coordinates for location matching
            self.city_coordinates = {
                'delhi': (28.6139, 77.2090),
                'new delhi': (28.6139, 77.2090),
                'mumbai': (19.0760, 72.8777),
                'bangalore': (12.9716, 77.5946),
                'bengaluru': (12.9716, 77.5946),
                'chennai': (13.0827, 80.2707),
                'kolkata': (22.5726, 88.3639),
                'hyderabad': (17.3850, 78.4867),
                'pune': (18.5204, 73.8567),
                'ahmedabad': (23.0225, 72.5714),
                'jaipur': (26.9124, 75.7873),
                'lucknow': (26.8467, 80.9462),
                'chandigarh': (30.7333, 76.7794),
                'gurgaon': (28.4595, 77.0266),
                'gurugram': (28.4595, 77.0266),
                'noida': (28.5355, 77.3910),
                'ghaziabad': (28.6692, 77.4538),
                'faridabad': (28.4089, 77.3178),
                'indore': (22.7196, 75.8577),
                'bhopal': (23.2599, 77.4126),
                'patna': (25.5941, 85.1376),
                'nagpur': (21.1458, 79.0882),
                'surat': (21.1702, 72.8311),
                'vadodara': (22.3072, 73.1812),
                'rajkot': (22.3039, 70.8022),
                'coimbatore': (11.0168, 76.9558),
                'madurai': (9.9252, 78.1198),
                'kochi': (9.9312, 76.2673),
                'thiruvananthapuram': (8.5241, 76.9366),
                'visakhapatnam': (17.6868, 83.2185),
                'vijayawada': (16.5062, 80.6480),
                'bhubaneswar': (20.2961, 85.8245),
                'raipur': (21.2514, 81.6296),
                'ranchi': (23.3441, 85.3096),
                'dehradun': (30.3165, 78.0322),
                'shimla': (31.1048, 77.1734),
                'jammu': (32.7266, 74.8570),
                'srinagar': (34.0837, 74.7973),
                'guwahati': (26.1445, 91.7362),
                'agartala': (23.8315, 91.2868)
            }
            
            print("Medical Voice Assistant initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            sys.exit(1)
    
    def extract_location_from_speech(self, text: str) -> Optional[Tuple[float, float]]:
        """Extract user location from speech text"""
        text_lower = text.lower()
        
        for city, coords in self.city_coordinates.items():
            if city in text_lower:
                print(f"📍 Detected location: {city.title()}")
                return coords
        
        return None
    
    def get_user_location(self) -> Optional[Tuple[float, float]]:
        """Get user location through voice input"""
        self.voice_processor.speak_response(
            "To find nearby hospitals, please tell me your current city or location. "
            "You can say 'skip' if you prefer not to share your location."
        )
        
        location_input = self.voice_processor.listen_to_speech()
        print(f"Location input: {location_input}")
        
        if "skip" in location_input.lower() or "not" in location_input.lower():
            return None
        
        return self.extract_location_from_speech(location_input)
    
    def get_symptoms_with_retry(self, max_retries: int = 3) -> str:
        """Get symptoms from user with retry mechanism for better accuracy"""
        
        # Initialize speech recognition with better settings (similar to test code)
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Adjust for ambient noise first (like in test code)
        print(" Adjusting for ambient noise... Please wait.")
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"Ambient noise level set to: {recognizer.energy_threshold}")
        except Exception as e:
            print(f"⚠️ Could not adjust for ambient noise: {e}")
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    prompt = (
                        "Please describe your symptoms clearly and in detail. "
                    )
                else:
                    prompt = (
                        f"Let me try again. This is attempt {attempt + 1} of {max_retries}. "
                        "Please speak slowly and clearly about your symptoms."
                    )
                
                print(f"\n Listening for symptoms (Attempt {attempt + 1}/{max_retries})...")
                self.voice_processor.speak_response(prompt)
                
                import time
                time.sleep(2) 

                print("Listening... (up to 10 seconds)")
                try:
                    with microphone as source:
                        # Listen with longer timeout and phrase limit
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
                    
                    print("Processing speech...")
                    symptom_text = recognizer.recognize_google(audio)
                    print(f"User input: '{symptom_text}'")
                    
                except sr.WaitTimeoutError:
                    print("Timeout - no speech detected")
                    if attempt < max_retries - 1:
                        self.voice_processor.speak_response(
                            "I didn't hear anything. Please try speaking again."
                        )
                    continue
                except sr.UnknownValueError:
                    print(" Could not understand the audio")
                    if attempt < max_retries - 1:
                        self.voice_processor.speak_response(
                            "I couldn't understand what you said. Please speak more clearly."
                        )
                    continue
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                    if attempt < max_retries - 1:
                        self.voice_processor.speak_response(
                            "There was a connection issue. Let me try again."
                        )
                    continue
                
                # Check if we got meaningful input
                if (symptom_text and 
                    len(symptom_text.strip()) > 5 and 
                    "sorry" not in symptom_text.lower() and 
                    "couldn't understand" not in symptom_text.lower() and
                    "could not understand" not in symptom_text.lower()):
                    
                    # Confirm what we heard
                    confirmation = f"I heard you say: {symptom_text}. Is this correct? Please say yes or no."
                    self.voice_processor.speak_response(confirmation)
                    
                    try:
                        print("Waiting for confirmation...")
                        with microphone as source:
                            audio = recognizer.listen(source, timeout=8, phrase_time_limit=3)
                        confirmation_response = recognizer.recognize_google(audio)
                        print(f" Confirmation: {confirmation_response}")
                        
                        if "yes" in confirmation_response.lower() or "correct" in confirmation_response.lower():
                            return symptom_text
                        else:
                            print("User indicated the transcription was incorrect, retrying...")
                            continue
                    except Exception as e:
                        print(f"⚠️ Confirmation failed: {e}, assuming transcription was correct")
                        return symptom_text
                else:
                    print("Poor quality input detected, retrying...")
                    if attempt < max_retries - 1:
                        self.voice_processor.speak_response(
                            "I didn't catch that clearly. Please make sure you're in a quiet place and speak directly into your microphone."
                        )
                    continue
                    
            except Exception as e:
                print(f" Error during speech recognition attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    self.voice_processor.speak_response(
                        "There was an issue with speech recognition. Let me try again."
                    )
                continue
        
        # If all attempts failed, ask for manual input or provide fallback
        self.voice_processor.speak_response(
            "I'm having trouble understanding your speech after multiple attempts. "
            "This might be due to background noise, microphone issues, or internet connectivity problems. "
            "Please make sure you're in a quiet environment, speaking directly into your microphone, "
            "and have a stable internet connection."
        )
        
        return ""
    
    def check_emergency(self) -> bool:
        """Check if this is an emergency situation"""
        self.voice_processor.speak_response(
            "Is this an emergency situation? Please say yes or no."
        )
        
        emergency_input = self.voice_processor.listen_to_speech()
        print(f" Emergency input: {emergency_input}")
        
        return "yes" in emergency_input.lower() or "emergency" in emergency_input.lower()
    
    def provide_comprehensive_guidance(self, symptoms: List[str], specialist: str, 
                                     confidence: float, hospitals_df, user_location: Optional[Tuple[float, float]]):
        """Provide comprehensive medical guidance"""
        
        # Specialist recommendation
        confidence_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        specialist_response = (
            f"Based on your symptoms: {', '.join(symptoms)}, "
            f"I recommend consulting a {specialist} with {confidence_text} confidence. "
        )
        
        # Hospital recommendations
        if not hospitals_df.empty:
            hospital_count = min(3, len(hospitals_df))
            hospital_response = f"I found {hospital_count} suitable hospitals for you: "
            
            for idx, (_, hospital) in enumerate(hospitals_df.head(3).iterrows()):
                distance_info = ""
                if 'distance_km' in hospital and user_location:
                    distance_info = f", which is {hospital['distance_km']:.1f} kilometers away"
                
                hospital_response += (
                    f"{idx + 1}. {hospital['name']} in {hospital['location']}{distance_info}. "
                    f"Rating: {hospital['rating']} out of 5. "
                    f"Contact: {hospital['phone']}. "
                )
        else:
            hospital_response = "I couldn't find specific hospitals matching your criteria. I recommend searching for general hospitals in your area. "
        
        # Emergency advice
        emergency_keywords = ['chest pain', 'shortness of breath', 'severe pain', 'bleeding', 'unconscious']
        if any(keyword in ' '.join(symptoms).lower() for keyword in emergency_keywords):
            emergency_response = "⚠️ IMPORTANT: Your symptoms may require immediate medical attention. Please consider visiting the nearest emergency room or calling emergency services. "
        else:
            emergency_response = ""
        
        # Complete response
        complete_response = emergency_response + specialist_response + hospital_response
        
        # Add general advice
        complete_response += (
            "Please remember that this is general guidance only. "
            "Always consult with a qualified healthcare professional for proper diagnosis and treatment. "
            "If your symptoms worsen or you feel this is an emergency, seek immediate medical help."
        )
        
        return complete_response
    
    def run_medical_consultation(self):
        """Main consultation flow"""
        try:
            # Welcome message
            welcome_message = (
                "Welcome to your Medical Voice Assistant! "
                "I'm here to help you find the right specialist and hospital based on your symptoms. "
                "Make sure you're in a quiet place for better speech recognition."
            )
            
            print(welcome_message)
            self.voice_processor.speak_response(welcome_message)
            
            # Get symptoms from user with improved retry mechanism
            symptom_text = self.get_symptoms_with_retry()
            
            if not symptom_text:
                self.voice_processor.speak_response(
                    "I wasn't able to understand your symptoms clearly. "
                    "Please try again later or consult a healthcare professional directly."
                )
                return
            
            # Extract symptoms using NLP
            print("Analyzing symptoms...")
            symptoms = self.medical_nlp.extract_symptoms(symptom_text)
            print(f"📋 Extracted symptoms: {symptoms}")
            
            if not symptoms:
                # Fallback: use the raw text if no specific symptoms extracted
                symptoms = [symptom_text]
            
            # Get specialist recommendation
            print("Finding specialist recommendation...")
            specialist, confidence = self.specialist_recommender.recommend_specialist(symptoms)
            print(f"Recommended specialist: {specialist} (confidence: {confidence:.2f})")
            
            # Get user location
            user_location = self.get_user_location()
            
            # Check if emergency
            is_emergency = self.check_emergency()
            
            # Find suitable hospitals
            print("Finding suitable hospitals...")
            if is_emergency:
                hospitals = self.hospital_matcher.find_emergency_hospitals(
                    user_location, max_distance=100
                )
            else:
                # Map specialist to medical specialty
                specialty_mapping = {
                    'Cardiologist': 'cardiology',
                    'Dermatologist': 'dermatology', 
                    'Gastroenterologist': 'gastroenterology',
                    'Neurologist': 'neurology',
                    'Pulmonologist': 'pulmonology',
                    'Orthopedist': 'orthopedics',
                    'Psychiatrist': 'psychiatry',
                    'General Practitioner': 'general medicine'
                }
                
                specialty = specialty_mapping.get(specialist, 'general medicine')
                hospitals = self.hospital_matcher.get_comprehensive_recommendation(
                    specialty, user_location, None, is_emergency  # Removed insurance parameter
                )
            
            print(f"Found {len(hospitals)} hospitals")
            
            # Provide comprehensive guidance
            guidance = self.provide_comprehensive_guidance(
                symptoms, specialist, confidence, hospitals, user_location
            )
            
            print(f"\n📢 Response: {guidance}")
            self.voice_processor.speak_response(guidance)
            
            # Ask if user wants more information
            self.voice_processor.speak_response(
                "Would you like more information about any specific hospital, "
                "or do you have any other questions? Say 'yes' for more information or 'no' to end."
            )
            
            more_info = self.voice_processor.listen_to_speech()
            
            if "yes" in more_info.lower():
                self.voice_processor.speak_response(
                    "You can find detailed information about these hospitals online, "
                    "or call the numbers I provided. Stay safe and take care!"
                )
            
            self.voice_processor.speak_response(
                "Thank you for using the Medical Voice Assistant. "
                "Please remember to consult with healthcare professionals for proper medical advice. "
                "Take care and get well soon!"
            )
            
        except KeyboardInterrupt:
            print("\nSession ended by user")
            self.voice_processor.speak_response("Session ended. Take care!")
        except Exception as e:
            print(f" Error during consultation: {str(e)}")
            self.voice_processor.speak_response(
                "I encountered an error. Please try again or consult a healthcare professional directly."
            )
    
    def run_interactive_mode(self):
        """Run in interactive mode with menu options"""
        while True:
            try:
                menu_message = (
                    "Welcome to Medical Voice Assistant! "
                    "Say 'symptoms' to describe your symptoms and get recommendations, "
                    "say 'hospitals' to find hospitals directly, "
                    "or say 'quit' to exit."
                )
                
                print("\n" + "="*50)
                print("MEDICAL VOICE ASSISTANT - MAIN MENU")
                print("="*50)
                print("Options:")
                print("1. Say 'symptoms' - Full consultation with symptom analysis")
                print("2. Say 'hospitals' - Direct hospital search")
                print("3. Say 'quit' - Exit application")
                print("="*50)
                
                self.voice_processor.speak_response(menu_message)
                user_choice = self.voice_processor.listen_to_speech()
                print(f" User choice: {user_choice}")
                
                if "quit" in user_choice.lower() or "exit" in user_choice.lower():
                    self.voice_processor.speak_response("Thank you for using Medical Voice Assistant. Goodbye!")
                    break
                elif "symptoms" in user_choice.lower() or "consultation" in user_choice.lower():
                    self.run_medical_consultation()
                elif "hospitals" in user_choice.lower() or "hospital" in user_choice.lower():
                    self.hospital_voice_interface.interactive_hospital_search()
                else:
                    self.voice_processor.speak_response(
                        "I didn't understand your choice. Please say 'symptoms', 'hospitals', or 'quit'."
                    )
            
            except KeyboardInterrupt:
                print("\n👋 Exiting Medical Voice Assistant")
                break
            except Exception as e:
                print(f" Error in interactive mode: {str(e)}")
                self.voice_processor.speak_response("An error occurred. Please try again.")

def main():
    print("Starting Medical Voice Assistant...")
    
    try:
        # Initialize the assistant
        assistant = MedicalVoiceAssistant()
        
        # Run in interactive mode
        assistant.run_interactive_mode()
        
    except Exception as e:
        print(f"Failed to start Medical Voice Assistant: {str(e)}")
        print("Please make sure all required dependencies are installed:")
        print("- pip install speech_recognition pyttsx3 gtts pygame")
        print("- pip install transformers torch pandas scikit-learn geopy")
        print("- pip install numpy")

if __name__ == "__main__":
    main()