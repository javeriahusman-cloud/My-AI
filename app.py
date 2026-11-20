import time
from flask import Flask, render_template, request, jsonify
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, template_folder='templates')

# Initialize the Gemini client securely using your .env variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # List of stable production models to try automatically in sequence if one is overloaded
    candidate_models = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]
    last_error = None
    
    for model_name in candidate_models:
        max_retries = 2
        delay = 1  # 1-second initial retry delay
        
        for attempt in range(max_retries):
            try:
                # Stateless generation ensures maximum reliability and seamless fallback routing
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_message
                )
                if response and response.text:
                    return jsonify({"response": response.text})
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # If it's a 503/high-demand error, retry shortly; otherwise try the next model
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                break  # Break out of retry loop to try the next backup model
                
    # If all backup models are exhausted, return a clean message instead of crashing
    return jsonify({"error": f"Servers are experiencing high traffic. Please try again in a moment."}), 500

@app.route("/reset", methods=["POST"])
def reset():
    # Stateless chat resets cleanly instantly
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)