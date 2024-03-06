from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer, util
import sounddevice as sd
import numpy as np
import tkinter as tk
import threading
import time
import sys
import serial
import pyttsx3

# Initialize Vosk speech recognition model
model_vosk = Model("vosk_model")
recognizer = KaldiRecognizer(model_vosk, 16000)

# Initialize Hugging Face Transformers Question Answering model
model_qa = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_qa, tokenizer=model_qa)

# Load model & tokenizer for Question-Answering
model = AutoModelForQuestionAnswering.from_pretrained(model_qa)
tokenizer = AutoTokenizer.from_pretrained(model_qa)

# Initialize pre-treined sentence embedding model
model_sentence = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class TextRedirector:
    def __init__(self, terminal):
        self.terminal = terminal

    def write(self, text):
        self.terminal.insert(tk.END, text)
        self.terminal.see(tk.END)  # Scroll to the end

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polibatam Direction Guide")

        # Initialize pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 90)  # Speed of speech (words per minute)
        self.engine.setProperty("volume", 1.0)
        
        # Load and set the logo as the window icon
        logo_image = tk.PhotoImage(file="image/logo.png")
        root.iconphoto(True, logo_image)
        
        self.recording = False
        self.recognition_thread = None
        
        #Create a button
        font_size = 16
        font = ("Times New Roman", font_size)

        self.start_stop_button = tk.Button(root, text="START",font=font, command=self.toggle_recognition)
        self.start_stop_button.pack(pady=10)
        self.start_stop_button.place(x=20, y=20, width=250, height=50)
        
        #Create the text display
        self.terminal_label = tk.Label(root, text="Terminal :")
        self.terminal_label.place(x=20, y=80)

        self.terminal_display = tk.Text(root, wrap=tk.WORD)
        self.terminal_display.pack(pady=10)
        self.terminal_display.place(x=20, y=100, width=250, height=330)

        #Create Word Error Rate text display
        self.wer_label = tk.Label(root, text="Word Error Rate :")
        self.wer_label.place(x=300, y=80)

        self.wer_display = tk.Text(root, wrap=tk.WORD)
        self.wer_display.pack(pady=10)
        self.wer_display.place(x=300, y=100, width=150, height=150)

        #Create Detection Speed text display
        self.time_label = tk.Label(root, text="Response time(s) :")
        self.time_label.place(x=300, y=260)

        self.time_display = tk.Text(root, wrap=tk.WORD)
        self.time_display.pack(pady=10)
        self.time_display.place(x=300, y=280, width=150, height=150)

        #Create Q&A text display
        self.q_label = tk.Label(root, text="Question :")
        self.q_label.place(x=500, y=80)

        self.q_display = tk.Text(root, wrap=tk.WORD)
        self.q_display.pack(pady=10)
        self.q_display.place(x=500, y=100, width=280, height=150)

        self.a_label = tk.Label(root, text="Answer :")
        self.a_label.place(x=500, y=260)

        self.a_display = tk.Text(root, wrap=tk.WORD)
        self.a_display.pack(pady=10)
        self.a_display.place(x=500, y=280, width=280, height=150)

    def toggle_recognition(self):
        if self.recording:
            self.recording = False
            self.clear_terminal_display
            self.start_stop_button.config(text="START")

        else:
            self.recording = True
            self.start_stop_button.config(text="STOP")
            self.recognition_thread = threading.Thread(target=self.recognition_loop)
            self.recognition_thread.start()
    
    def recognition_loop(self):
        # Initialize Serial Port
        # port = serial.Serial('/dev/ttyACM0',1000000)
        
        # Initialize variables
        max_attempts = 5
        attempts = 0
        total_wer_percentage = 0

        #Print out the from terminal
        sys.stdout = TextRedirector(self.terminal_display)

        # Define the initial context for question answering
        context = ('Immigration office is at Teluk Tering Engku Putri Road Number 3'
                   "Airport is at Hang Nadim Road Number 1 Nongsa Sub-District and Mayor's office is at Teluk Tering Engku Putri Road Number 1 "
                   'Nearest Hospital is at Taman Baloi Raja Haji Fisabilillah Road  and gas station is at Baloi Permai Ahmad Yani Road')


        while self.recording and attempts < max_attempts:
            print(f"Attempt {attempts + 1}")

            recognizer.Reset() #Reset before the next attempt

            # Record audio using sounddevice
            duration = 5
            sample_rate = 16000
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
                
            # Convert the recorded audio to bytes for Vosk
            audio_data = np.array(recording, dtype=np.int16).tobytes()
                
            # Define the correct reference text and calculate embeddings reference sentences
            reference_input = ["Where is the immigration office","Where is the mayor's office","Where is the airport",
                                "Where is the nearest gas station","Where is the nearest hospital"]
            reference_embeddings = model_sentence.encode(reference_input, convert_to_tensor=True)
                
            # Start measuring time 
            start_time = time.time()

            if recognizer.AcceptWaveform(audio_data):
                end_time = time.time()
                delta_time = end_time - start_time
                    
                recognition = recognizer.Result()
                text = recognition.split(":")
                recognized_text = text[1].strip().replace('"', '').replace('}', '').replace('\n', '').replace(' ?', '?')
                recognized_text = recognized_text[:1].upper() + recognized_text[1:]
                print("Input:", recognized_text)
                    
                # Calculate embeddings for recognized query 
                recognized_embedding = model_sentence.encode([recognized_text], convert_to_tensor=True)

                # Calculate cosine similarities between recognized embedding and reference embeddings
                cosine_similarities = util.pytorch_cos_sim(recognized_embedding, reference_embeddings)

                # Find the index of the closest reference based on highest similarity
                closest_reference_index = np.argmax(cosine_similarities)
                closest_reference_text = reference_input[closest_reference_index]
                    
                # Calculate Substitution, Deletion, and Insertion errors
                recognized_words = recognized_text.split()
                reference_words = closest_reference_text.split()
                
                S = sum(1 for r, h in zip(reference_words, recognized_words) if r != h)
                D = max(0, len(reference_words) - len(recognized_words))
                I = max(0, len(recognized_words) - len(reference_words))

                # Calculate Word Error Rate
                N = len(reference_words)
                single_wer_percentage = (S + D + I) / N
                total_wer_percentage += single_wer_percentage
                    
                print("S :",S)
                print("D :",D)
                print("I :",I)
                print("N :",N)
                    
                self.wer_display.delete("1.0", "end")
                self.wer_display.insert("1.0", f"{single_wer_percentage:.6f}")
                self.time_display.delete("1.0", "end")
                self.time_display.insert("1.0", f"{delta_time:.2f} seconds")
                
                # Use the existing pipeline for question answering
                if recognized_text.strip():
                    QA_input = {'question': recognized_text,'context': context}
                    res = qa_pipeline(QA_input) 
                    self.q_display.delete("1.0", "end")
                    self.q_display.insert("1.0", recognized_text + "?")
                else:
                    self.q_display.delete("1.0", "end")
                    self.q_display.insert("1.0", "No input to use as a question.")
                    self.a_display.delete("1.0", "end")
                    
                if res["score"] > 0.1:
                    self.a_display.delete("1.0", "end")
                    self.a_display.insert("1.0", res['answer'])
                    # port.write(bytes([1]))
                    self.engine.say(res['answer'])
                    print("Score :", res['score'])
                    print("Output :", res['answer'])
                    self.engine.runAndWait()
                else:
                    self.a_display.delete("1.0", "end")
                    self.a_display.insert("1.0", "No answer found for the question.")
                    # port.write(bytes([2]))
                    self.engine.say("No answer found for the question.")               
                    self.engine.runAndWait()
            attempts += 1
            print("\n")

        # Calculate the Word Error Rate percentage after max_attempts
        wer_percentage = total_wer_percentage / max_attempts            
        print("Total Word Error Rate:", wer_percentage)
        print("\n")

        self.recording = False            
        self.start_stop_button.config(text="START")
        
    def clear_terminal_display(self):
        self.terminal_display.delete("1.0", tk.END)  # Delete all text
        self.wer_display.delete("1.0", tk.END)
        self.time_display.delete("1.0", tk.END)
        self.q_display.delete("1.0", tk.END)
        self.a_display.delete("1.0", tk.END)

# Create the Tkinter window
root = tk.Tk()
app = SpeechRecognitionApp(root)

# Set the dimensions of the window
window_width = 800
window_height = 460
windows_dimensions = f"{window_width}x{window_height}"
root.geometry(windows_dimensions)

root.mainloop()
