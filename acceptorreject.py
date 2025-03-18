import cv2
import face_recognition
import whisper
import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input
class LoanRequest(BaseModel):
    transcription: str


# ====== Setup Directories ======
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

# ====== Filenames ======
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(output_dir, f"recording_{timestamp}.avi")
audio_filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
transcription_filename = os.path.join(output_dir, f"transcription_{timestamp}.txt")

# ====== Initialize Whisper Model ======
model = whisper.load_model("base")

# ====== Audio Setup ======
# ====== Audio Setup (Optimized) ======
CHUNK = 4096  # Larger chunk to reduce interruptions
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Lower rate to prevent lag and improve speech capture
audio = pyaudio.PyAudio()

def record_audio():
    """Record audio smoothly in a separate thread."""
    print("[INFO] Recording audio...")

    # Start audio stream with optimized settings
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    
    while recording:
        try:
            # Read audio with exception handling for overflow
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"[ERROR] Audio Buffer Overflow: {e}")

    # Stop the stream once done
    stream.stop_stream()
    stream.close()
    
    # Save audio to file
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"[INFO] Audio saved as: {audio_filename}")


# ====== Video Setup ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 10.0, (320, 240))

print("[INFO] Please look into the camera for face enrollment...")
ret, frame = cap.read()
if not ret:
    print("[ERROR] Could not access the camera.")
    exit()
# ====== Face Enrollment ======
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_frame)

if len(face_locations) == 0:
    print("[ERROR] No face detected for enrollment. Exiting...")
    cap.release()
    exit()

enrolled_face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
print("[INFO] Face enrollment successful.")

# ====== Start Audio Thread ======
recording = True
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

print("[INFO] Recording started... Press 'q' to stop.")
warning_displayed = False

# ====== Video + Face Recognition Loop ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    current_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(current_face_encodings) > 0:
        match = face_recognition.compare_faces([enrolled_face_encoding], current_face_encodings[0])

        if not match[0]:
            cv2.putText(frame, "WARNING: Different Person Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not warning_displayed:
                print("[WARNING] Different person detected!")
                warning_displayed = True
        else:
            warning_displayed = False

    # Draw face rectangles
    for (top, right, bottom, left) in face_locations:
        color = (0, 255, 0) if match[0] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Show real-time video feed
    cv2.imshow("Recording (Press 'q' to stop)", frame)
    
    # Save frame to video
    out.write(frame)

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Stop Recording ======
recording = False
audio_thread.join()
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Video saved as: {video_filename}")

# ====== Transcribe Audio ======
print("[INFO] Transcribing audio...")
result = model.transcribe(audio_filename)

# Save transcription with UTF-8 encoding
with open(transcription_filename, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"[INFO] Transcription saved as: {transcription_filename}")
print("Transcription:", result["text"])


# ====== Loan Eligibility Evaluation ======
def extract_income(text):
    """Extract income from transcription."""
    # Handle formats: "Rs. 50000", "rupees 50000", "50,000 rupees", "50k"
    income_patterns = [
        r"(?i)rs\.?\s*(\d{1,3}(?:,\d{3})*|\d+)",  # "Rs. 50,000"
        r"(?i)rupees\s*(\d{1,3}(?:,\d{3})*|\d+)",  # "rupees 50,000"
        r"(?i)(\d{1,3}(?:,\d{3})*|\d+)\s*rupees",  # "50,000 rupees"
        r"(?i)(\d{1,3}(?:,\d{3})*|\d+)k",  # "50k"
    ]

    for pattern in income_patterns:
        match = re.search(pattern, text)
        if match:
            income_str = match.group(1).replace(",", "")  # Remove commas
            return int(income_str) * 1000 if "k" in pattern else int(income_str)

    return None

def extract_age(text):
    """Extract age from transcription."""
    age_patterns = [
        r"I am (\d{1,3}) years old",
        r"I'm (\d{1,3})",
        r"age (\d{1,3})"
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 100:
                return age

    return None

def evaluate_loan_eligibility(transcription):
    """Evaluate loan eligibility based on extracted info."""
    print("[INFO] Evaluating Loan Eligibility...")

    age = extract_age(transcription)
    income = extract_income(transcription)
    reasons = []

    # Eligibility check
    if age is None:
        reasons.append("Age not mentioned.")
    elif age < 18:
        reasons.append("Applicant must be at least 18 years old.")

    if income is None:
        reasons.append("Income not mentioned.")
    elif income < 15000:
        reasons.append("Income too low.")
    elif 15000 <= income < 25000:
        reasons.append("More financial info needed.")

    # Final decision
    if not reasons:
        decision = "✅ Approved"
    elif "Income too low." in reasons or "Applicant must be at least 18 years old." in reasons:
        decision = "❌ Rejected"
    else:
        decision = "🔄 More Info Needed"

    # Show results
    print("Final Decision:", decision)
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print("-", reason)

    return decision, reasons


# ====== Call the function after transcription ======
decision, reasons = evaluate_loan_eligibility(result["text"])


print("\nFinal Decision:", decision)
if reasons:
    print("Reasons:")
    for reason in reasons:
        print("-", reason)

print("[INFO] All tasks completed!")

@app.post("/evaluate/")
def evaluate_loan(request: LoanRequest):
    decision, reasons = evaluate_loan_eligibility(request.transcription)
    return {"decision": decision, "reasons": reasons}
