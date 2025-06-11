import streamlit as st
import threading
import time
from speech_rec import VoiceProcessor
from specialist import SpecialistRecommender
from medical import MedicalNLP

class MedicalChatbot:
    def __init__(self):
        self.voice_processor = VoiceProcessor()
        self.nlp_processor = MedicalNLP()
        self.recommender = SpecialistRecommender()
    
    def process_patient_input(self, text_input=None):
        if text_input:
            user_input = text_input
        else:
            user_input = self.voice_processor.listen_to_speech()
        
        if "sorry" in user_input or "couldn't understand" in user_input:
            return user_input, None, None
        
        symptoms = self.nlp_processor.extract_symptoms(user_input)
        specialist, confidence = self.recommender.recommend_specialist(symptoms)
        
        return user_input, symptoms, (specialist, confidence)
    
    def generate_response(self, symptoms, specialist_info):
        specialist, confidence = specialist_info
        
        if confidence > 0.5:
            response = f"Based on your symptoms: {', '.join(symptoms)}, "
            response += f"I recommend consulting a {specialist}. "
            response += f"Confidence level: {confidence:.1%}"
        else:
            response = "Based on your symptoms, I recommend starting with a General Practitioner "
            response += "who can provide initial assessment and referrals if needed."
        
        response += "\n\nImportant: This is AI assistance only. Please consult healthcare professionals for proper diagnosis."
        
        return response

def main():
    st.title("AI Medical Symptom Advisor")
    st.write("Describe your symptoms and get specialist recommendations")
    
    chatbot = MedicalChatbot()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Voice Input"):
            with st.spinner("Listening..."):
                user_input, symptoms, specialist_info = chatbot.process_patient_input()
                
                if symptoms and specialist_info:
                    response = chatbot.generate_response(symptoms, specialist_info)
                    
                    # Add to history
                    st.session_state.conversation_history.append({
                        'user': user_input,
                        'symptoms': symptoms,
                        'response': response
                    })
                    
                    chatbot.voice_processor.speak_response(response)
    
    with col2:
        text_input = st.text_area("Or type your symptoms:")
        if st.button("Text Input") and text_input:
            user_input, symptoms, specialist_info = chatbot.process_patient_input(text_input)
            
            if symptoms and specialist_info:
                response = chatbot.generate_response(symptoms, specialist_info)
                
                st.session_state.conversation_history.append({
                    'user': user_input,
                    'symptoms': symptoms,
                    'response': response
                })
    
    for chat in st.session_state.conversation_history:
        st.write("**You:**", chat['user'])
        st.write("**Detected Symptoms:**", ', '.join(chat['symptoms']))
        st.write("**AI Advisor:**", chat['response'])
        st.write("---")

if __name__ == "__main__":
    main()