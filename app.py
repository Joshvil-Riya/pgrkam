# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import csv
import requests 
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from chat import get_response
import os

from flask import Flask, render_template, request, redirect, url_for,Markup
import sounddevice as sd
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr




import joblib
import numpy as np;
import pandas as pd;
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]
import numpy as np;
import pandas as pd;
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import random



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np;
import pandas as pd;
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

import speech_recognition as sr
# ------------------------------------ FLASK APP -------------------------------------------------
import random
import string


from flask import Flask, render_template, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
import json
import numpy as np

# Generate three random alphabets
random_alphabets = ''.join(random.choices(string.ascii_uppercase, k=3))

# Generate three random integers
random_integers = ''.join(random.choices(string.digits, k=3))

# Combine alphabets and integers
random_combination = random_alphabets + random_integers

print(random_combination)



app = Flask(__name__)

# render home page

# Function to initialize or load chat history
def load_chat_history():
    chat_history = []
    file_path = 'chat_history.csv'

    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                chat_history.append({'random_combination':[0],'user_message': row[1], 'bot_response': row[2]})

    return chat_history

def load_chat_history2(target_combination):
    chat_history = []
    file_path = 'chat_history.csv'

    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                # Assuming the 'random_combination' field is in the second column (index 1)
                if len(row) > 1 and row[0] == target_combination:
                    chat_history.append({'random_combination': [0], 'user_message': row[1], 'bot_response': row[2]})

    return chat_history

# Function to save a new message to the chat history
def save_to_chat_history(random_combination,user_message, bot_response):
    file_path = 'chat_history.csv'

    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([random_combination,user_message, bot_response])
#@ app.route('/')
#def home():
#    title = 'Coalbot'
#    return render_template('index4.html', title=title)

# render crop recommendation form page

@app.route('/')
def home():
    return render_template('home_1.html') 




@app.route('/chatbot1',methods=['POST','GET'])
def chatbot1():
    return render_template('chatbot.html')

'''#@app.route('/')
#def home():
#    return render_template('index5.html') '''

@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('services_page.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               



                          
                     # print(value1[0:])
    
    
    
    

              
              # int_features3[0]==12345 and int_features3[1]==12345:
               #                      return render_template('index.html')
        
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login44.html')

                      



@app.route('/chatboat', methods=['GET', 'POST'])
def chatboat():
   
    return render_template('login44.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']
    # You can replace the next line with a call to an AI model for generating responses
    bot_response ,predicted_tag=get_response(user_message)

    #save_to_chat_history(random_combination,user_message, bot_response)

    #chat_history = load_chat_history2(random_combination)

  #  print("the random generated result is ",chat_history)





    # Extract 'bot_response' value
    #bot_response_value = chat_history[0]['bot_response']

    
    from gtts import gTTS
    from pydub import AudioSegment
    import pygame
    import os

    text = str(bot_response)
    language = 'en'

    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)

        # Save the converted audio in a file
        tts.save("output.mp3")
        print("Audio file saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

    # Play the saved audio file using pygame
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load("output.mp3")

        # Play the audio file
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick value as needed

        # Close the mixer
        pygame.mixer.quit()

    except Exception as e:
        print(f"Error playing audio: {e}")
    

    from translate import Translator

    def translate_text(text, target_language):
        translator= Translator(to_lang=target_language)
        translation = translator.translate(text)
        return translation
   
    text_to_translate =bot_response
    # Translate to Kannada
    english_translation=text_to_translate
    pnb_translation = translate_text(text_to_translate, "pa")
    print(f"pnb: {pnb_translation}")



    # Print the result
    #print("this is the text response for present question",loaded_data_yield)
    #return render_template('index.html', user_message=user_message, bot_response=bot_response)
    save_to_chat_history(random_combination,user_message, bot_response=text_to_translate)

    chat_history = load_chat_history2(random_combination)

    #bot_response_value = chat_history[0]['bot_response']

    return render_template('chatbot.html', user_message=user_message, bot_response=text_to_translate, chat_history=chat_history,pnb_translation=pnb_translation)


@app.route('/ask2', methods=['POST'])
def ask2():




    def speech_to_text_for_5_seconds():
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Use the microphone as the audio source
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
            
            print("Listening for 5 seconds. Please speak...")
            try:
                # Listen to the microphone for 5 seconds
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Recognize speech using Google Web Speech API
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"Recognized Text: {text}")
                except sr.UnknownValueError:
                    print("Sorry, I could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Request error from Google Speech Recognition: {e}")
            
            except sr.WaitTimeoutError:
                print("No speech detected within the timeout period.")

        return     text                 

#if __name__ == "__main__":
    

    user_message = speech_to_text_for_5_seconds()
    # You can replace the next line with a call to an AI model for generating responses
    bot_response ,predicted_tag=get_response(user_message)

    #save_to_chat_history(random_combination,user_message, bot_response)

    #chat_history = load_chat_history2(random_combination)

  #  print("the random generated result is ",chat_history)





    # Extract 'bot_response' value
    #bot_response_value = chat_history[0]['bot_response']

    
    from gtts import gTTS
    from pydub import AudioSegment
    import pygame
    import os

    text = str(bot_response)
    language = 'en'

    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)

        # Save the converted audio in a file
        tts.save("output.mp3")
        print("Audio file saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

    # Play the saved audio file using pygame
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load("output.mp3")

        # Play the audio file
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick value as needed

        # Close the mixer
        pygame.mixer.quit()

    except Exception as e:
        print(f"Error playing audio: {e}")
    

    from translate import Translator

    def translate_text(text, target_language):
        translator= Translator(to_lang=target_language)
        translation = translator.translate(text)
        return translation
   
    text_to_translate =bot_response
    # Translate to Kannada
    english_translation=text_to_translate
    pnb_translation = translate_text(text_to_translate, "pa")
    print(f"pnb: {pnb_translation}")



    # Print the result
    #print("this is the text response for present question",loaded_data_yield)
    #return render_template('index.html', user_message=user_message, bot_response=bot_response)
    save_to_chat_history(random_combination,user_message, bot_response=text_to_translate)

    chat_history = load_chat_history2(random_combination)

    #bot_response_value = chat_history[0]['bot_response']

    return render_template('chatbot.html', user_message=user_message, bot_response=text_to_translate, chat_history=chat_history,pnb_translation=pnb_translation,english_translation=english_translation)
# Load the trained Random Forest model
model = joblib.load('random_forest_job_model_upsampled.pkl')

# Mapping dictionaries (must match training)
gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
qual_dict = {'10th':0, '12th':1, 'Diploma':2, 'Graduate':3, 'Postgrad':4}
job_dict = {v:k for k,v in {'Lab Assistant':0, 'Security Guard':1, 'Sales Executive':2,
                            'School Teacher':3, 'Delivery Boy':4, 'Junior Engineer':5,
                            'Data Entry Operator':6, 'Clerk':7, 'Customer Support':8, 'Apprentice':9}.items()}


@app.route('/input_form', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        # Get form data
        applicant_id = request.form['Applicant_ID']
        age = int(request.form['Age'])
        gender = request.form['Gender']
        qualification = request.form['Qualification_Level']
        marks = float(request.form['Marks_Percentage'])
        experience = int(request.form['Experience_Years'])
        driving = int(request.form.get('Driving',0))
        sales = int(request.form.get('Sales',0))
        typing = int(request.form.get('Typing',0))
        computer = int(request.form.get('Computer',0))
        govt_score = int(request.form.get('Govt_Exam_Score',0))
        english = int(request.form.get('English_Proficiency',0))
        computer_prof = int(request.form.get('Computer_Proficiency',0))
        physical = int(request.form.get('Physical_Fitness',0))

        # Convert categorical to numeric
        gender_num = gender_dict[gender]
        qual_num = qual_dict[qualification]

        # Prepare data for prediction
        X_input = [[age, gender_num, qual_num, marks, experience, driving, sales, typing, computer,
                    govt_score, english, computer_prof, physical]]

        pred_num = model.predict(X_input)[0]
        pred_job = job_dict[pred_num]

        # Save input + prediction to CSV
        record = {
            'Applicant_ID': applicant_id,
            'Age': age,
            'Gender': gender,
            'Qualification_Level': qualification,
            'Marks_Percentage': marks,
            'Experience_Years': experience,
            'Driving': driving,
            'Sales': sales,
            'Typing': typing,
            'Computer': computer,
            'Govt_Exam_Score': govt_score,
            'English_Proficiency': english,
            'Computer_Proficiency': computer_prof,
            'Physical_Fitness': physical,
            'Predicted_Job': pred_job
        }

        if os.path.exists('database.csv'):
            df_db = pd.read_csv('database.csv')
            df_db = pd.concat([df_db, pd.DataFrame([record])], ignore_index=True)
        else:
            df_db = pd.DataFrame([record])
        df_db.to_csv('database.csv', index=False)

        return render_template('result.html', prediction=pred_job, applicant_id=applicant_id)

    return render_template('input_form.html', gender_options=list(gender_dict.keys()),
                           qualification_options=list(qual_dict.keys()))





# Load CSV
df = pd.read_csv("database.csv")

# Mock data for time-series and other charts (since CSV may not have time/demographics; adapt as needed)
# For real implementation, derive from df if possible (e.g., add timestamps)
times = pd.date_range(start='2025-09-01', periods=24, freq='H').time
mock_page_views = np.random.randint(5000, 8000, 24).cumsum()
mock_unique_visitors = np.random.randint(2000, 4000, 24).cumsum()
mock_job_searches = np.random.randint(1000, 3000, 24).cumsum()

traffic_sources = {
    'Google Search': 35.4,
    'Direct Traffic': 28.7,
    'Social Media': 18.2,
    'Email Campaigns': 9.6,
    'Mobile App': 5.8,
    'Referrals': 2.3
}
traffic_df = pd.DataFrame(list(traffic_sources.items()), columns=['Source', 'Percentage'])

channel_data = {
    'Google Search': np.random.randint(2000, 3000, 7),
    'Direct Traffic': np.random.randint(1500, 2500, 7),
    'Social Media': np.random.randint(1000, 2000, 7),
    'Email Campaigns': np.random.randint(500, 1500, 7)
}
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

funnel_steps = {
    'Landing': 45672,
    'Browse Jobs': 38920,
    'View Job Details': 31136,
    'Create Profile': 24909,
    'Submit Job Apply': 18681,
    'Success': 12654
}
funnel_df = pd.DataFrame(list(funnel_steps.items()), columns=['Step', 'Users'])

#@app.route("/")
#def home():
#    return render_template("home2.html")

@app.route("/analytics")
def analytics():
    # Core metrics from CSV
    total_applications = len(df)
    gender_counts = df['Gender'].value_counts()
    qual_counts = df['Qualification_Level'].value_counts()
    job_counts = df['Predicted_Job'].value_counts()

    # Mock metrics to match screenshot style (adapt with real data if available)
    total_active_users = 45668
    page_views = 1284388
    job_applications = total_applications  # Use from CSV
    success_rate = 679  # Mock %
    avg_session_time = 430  # seconds
    click_through_rate = 322  # %
    mobile_traffic = 788  # %
    bounce_rate = 318  # %

    # Charts from CSV
    gender_fig = px.pie(df, names="Gender", title="Gender Distribution")
    gender_graph = json.dumps(gender_fig, cls=plotly.utils.PlotlyJSONEncoder)

    qual_fig = px.bar(qual_counts.reset_index(), x="index", y="Qualification_Level", title="Qualification Level Distribution")
    qual_graph = json.dumps(qual_fig, cls=plotly.utils.PlotlyJSONEncoder)

    job_fig = px.bar(job_counts.reset_index(), x="index", y="Predicted_Job", title="Predicted Job Distribution")
    job_graph = json.dumps(job_fig, cls=plotly.utils.PlotlyJSONEncoder)

    age_exp_fig = px.scatter(df, x="Age", y="Experience_Years", color="Predicted_Job", title="Age vs Experience by Predicted Job")
    age_exp_graph = json.dumps(age_exp_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # User Activity Over Time (mock line chart)
    activity_df = pd.DataFrame({
        'Time': times,
        'Page Views': mock_page_views,
        'Unique Visitors': mock_unique_visitors,
        'Job Searches': mock_job_searches
    })
    activity_fig = go.Figure()
    activity_fig.add_trace(go.Scatter(x=activity_df['Time'], y=activity_df['Page Views'], mode='lines', name='Page Views'))
    activity_fig.add_trace(go.Scatter(x=activity_df['Time'], y=activity_df['Unique Visitors'], mode='lines', name='Unique Visitors'))
    activity_fig.add_trace(go.Scatter(x=activity_df['Time'], y=activity_df['Job Searches'], mode='lines', name='Job Searches'))
    activity_fig.update_layout(title="User Activity Over Time")
    activity_graph = json.dumps(activity_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Traffic Sources Pie
    traffic_fig = px.pie(traffic_df, values='Percentage', names='Source', title="Traffic Sources Distribution")
    traffic_graph = json.dumps(traffic_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Weekly Channel Performance Bar
    weekly_fig = go.Figure()
    for channel, values in channel_data.items():
        weekly_fig.add_trace(go.Bar(x=days, y=values, name=channel))
    weekly_fig.update_layout(title="Weekly Channel Performance", barmode='group')
    weekly_graph = json.dumps(weekly_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # User Journey Funnel
    funnel_fig = px.funnel(funnel_df, x='Users', y='Step', title="User Journey Flow")
    funnel_graph = json.dumps(funnel_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("result2.html",
                           total_active_users=total_active_users,
                           page_views=page_views,
                           job_applications=job_applications,
                           success_rate=success_rate,
                           avg_session_time=avg_session_time,
                           click_through_rate=click_through_rate,
                           mobile_traffic=mobile_traffic,
                           bounce_rate=bounce_rate,
                           total_applications=total_applications,
                           gender_graph=gender_graph,
                           qual_graph=qual_graph,
                           job_graph=job_graph,
                           age_exp_graph=age_exp_graph,
                           activity_graph=activity_graph,
                           traffic_graph=traffic_graph,
                           weekly_graph=weekly_graph,
                           funnel_graph=funnel_graph)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
