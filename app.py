import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import google.generativeai as gen_ai

load_dotenv()

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# loading the saved models
with open('diabetes.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)
with open('heart_disease.pkl', 'rb') as file:
    heart_disease_model = pickle.load(file)

with st.sidebar:
    selected = option_menu(
        'Advanced Healthcare Solutions',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Personalized Treatment', 'Medical Image Analysis'],
        menu_icon='hospital-fill',
        icons=['house', 'activity', 'heart', 'gear', 'camera'],
        default_index=0
    )

# Set up Google Gemini-Pro AI model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Get the key from the environment
if GOOGLE_API_KEY:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-1.5-flash-latest')
else:
    st.error("Google API key is not found. Please check your .env file.")

def get_personalized_plan_from_gemini(weight, height, medical_history, activity_level, query):
    # Define the system prompt
    system_prompt = """
    You are a personalized healthcare assistant. Your role is to generate customized meal and workout plans based on the user's physical details and medical history. 
    Provide well-structured, actionable advice, considering the user's current health status and lifestyle. Be concise and informative.
    """
    
    # Define the user prompt with the user's inputs
    user_prompt = f"""
    User details:
    - Weight: {weight} kg
    - Height: {height} cm
    - Medical history: {medical_history}
    - Activity level: {activity_level}
    
    Generate a personalized meal plan and workout plan suitable for the user. Include daily meal recommendations and suggested exercises. Also, consider any medical conditions or health concerns provided in the medical history.
    """

    chat = model.start_chat(
        history = [
                {"role": "user", "parts": user_prompt},
                {"role": "model", "parts": system_prompt},
            ]
    )

    # Assuming there's a function to send prompts to the Gemini model and retrieve the output
    response = chat.send_message(query)
    return response.text


if selected == 'Home':
    st.markdown(
        """
        <div style="text-align: center;">
        <h2 style="color: #4285F4; font-size: 40px;">Your AI Health Companion! The Future of Personalized Care</h2>
        <p style="font-size: 20px;">Powered by Hack to the Future Hackathon</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Description for the features
    feature_1_desc = "Diabetes Prediction: Use machine learning to predict the likelihood of diabetes based on user inputs."
    feature_2_desc = "Heart Disease Prediction: A machine learning model to predict the possibility of heart disease."
    feature_3_desc = "Personalized Treatment: Get customized meal plans and workout routines based on your health data."
    feature_4_desc = "Medical Image Analysis: Upload medical images for an AI-powered analysis, providing insights into potential health concerns."

    # Image paths (update these with your actual image file paths)
    image_1_path = "./temp1.png"
    image_2_path = "./temp2.jpeg"
    image_3_path = "./temp3.jpeg"
    image_4_path = "./temp4.jpg"

    # Create two rows of features (2 in each row)
    col1, col2 = st.columns(2)

    # First row (feature 1 and 2)
    with col1:
        st.image(image_1_path, caption="Diabetes Prediction", use_column_width=True)
        st.write(feature_1_desc)

    with col2:
        st.image(image_2_path, caption="Heart Disease Prediction", use_column_width=True)
        st.write(feature_2_desc)

    # Second row (feature 3 and 4)
    col3, col4 = st.columns(2)

    with col3:
        st.image(image_3_path, caption="Personalized Treatment", use_column_width=True)
        st.write(feature_3_desc)

    with col4:
        st.image(image_4_path, caption="Medical Image Analysis", use_column_width=True)
        st.write(feature_4_desc)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)


if selected == 'Personalized Treatment':
    st.title("Personalized HealthCare Chatbot")

    # Collect user inputs
    weight = st.number_input('Enter your weight (kg)')
    height = st.number_input('Enter your height (cm)')
    medical_history = st.text_area("Enter your medical history and any existing conditions")
    activity_level = st.selectbox("Select your activity level", 
                                  ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
    
    query = st.text_input("Ask the Personalized HealthCare Chatbot...")

    # Submit button to get a response
    if st.button("Get Personalized Plan"):
        response = get_personalized_plan_from_gemini(weight, height, medical_history, activity_level, query)
        # Custom box styling using markdown
        if response:
            st.markdown(
                f"""
                <div style="
                    background-color: #black;
                    padding: 15px;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 20px;
                    ">
                    {response}
                </div>
                """, unsafe_allow_html=True
            )

if selected == 'Medical Image Analysis':
    st.title("Medical Image Analysis")

    system_prompt = """
You are a specialized AI-driven healthcare assistant tasked with providing a thorough analysis of medical images.
Your role is to carefully analyze the image and generate the following:
1. Detailed Analysis: Provide an in-depth explanation of the disease or condition identified from the medical image. 
Break down the key features visible in the image that led to your diagnosis.
2. Finding Reports: Summarize your findings based on the image analysis, including any abnormalities, signs, or patterns that indicate a particular health condition. 
Mention relevant details like the severity of the condition if visible in the image.
3. Recommendations and Next Steps: Based on the analysis, suggest immediate next steps for the user. 
This can include recommendations for further diagnostic tests, lifestyle adjustments, or areas requiring further medical attention.
4. Treatment Suggestions: Offer general treatment suggestions or preventive care advice for the identified condition. 
Mention that users should consult with a healthcare professional for confirmation and a personalized treatment plan.

Important Notes:
1. Scope of Response: Your analysis should be based solely on the visual information provided within the medical image.
2. Clarity of Image: You should only provide analysis based on the quality and clarity of the image. If the image is unclear, you should inform the user that the analysis may not be accurate due to poor image quality.
3. Disclaimer: Clearly state that the analysis is an AI-powered tool and is not a substitute for professional medical advice, diagnosis, or treatment.
4. Your Valuable Insights: Offer insights into the visual findings in the medical image, such as the presence of potential conditions

Always give reponse in the above 4 categories format explaining each section in point wise format.
"""

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Upload a medical image for analysis", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        submit_button = st.button('Analyze the medical Image')
        if submit_button:
            image_data = uploaded_file.getvalue()
            image_parts = [
                {
                    "mime_type" : "image/jpeg",
                    "data": image_data
                }
            ]
            prompt_parts = [
                image_parts[0],
                system_prompt,
            ]
            response = model.generate_content(prompt_parts)
            if response:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #black;
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid #ddd;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        margin-top: 20px;
                        ">
                        {response.text}
                    </div>
                    """, unsafe_allow_html=True
                )
