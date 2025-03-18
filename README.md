acceptorreject.py:

Video-Based Customer Interaction

Users record video responses instead of filling lengthy forms,
Basic facial verification ensures the same applicant continues, 
Ensures that the same person interacts throughout the session, preventing identity fraud,
records the customer interaction,
The system listens to the user’s answers through audio,
Converts the user’s spoken responses into text using Whisper AI,
Right after transcription, the system analyzes the user's answers (age, income, employment) and evaluates their loan eligibility.

loan_api.py:

A rule-based system evaluates loan eligibility using user responses and document data,
Provides instant feedback: ✅ Approved, ❌ Rejected (with reasons), 🔄 More Info Needed.
