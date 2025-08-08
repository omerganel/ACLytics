import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow cross-origin calls from your frontend

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/api/gpt", methods=["POST"])
def gpt_response():
    data = request.json

    required_fields = ['age', 'height', 'weight', 'sport', 'leftKneeAngle', 'rightKneeAngle', 'imageBase64']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required data"}), 400

    # Build the prompt
    prompt = f"""
Analyze the following athlete's biomechanical data and assess their ACL injury risk. Provide practical, clear advice in simple language.

Athlete:
- Age: {data['age']}
- Height: {data['height']}
- Weight: {data['weight']}
- Sport: {data['sport']}
- Left Knee Angle: {data['leftKneeAngle']}°
- Right Knee Angle: {data['rightKneeAngle']}°

The athlete also submitted an annotated image of their posture (not shown here). Use the data provided to identify any concerns and offer actionable recommendations.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Change to gpt-4 if you have access
            messages=[
                {
                    "role": "system",
                    "content": "You are a sports medical doctor specializing in ACL injuries, with a deep understanding of biomechanics, sports performance, and injury prevention. Your role is to evaluate athletes’ ACL injury risk through analysis of provided images (with visible joint angles and body positions) and personal data including age, height, weight, and primary sport. Make sure to refer to the user's specific personal data that they inputted when providing professional feedback, including their age, height, weight, and primary sport(mention each part of their inputted personal data in your response and provide information specifically tailored to that certain aspect of their personal data). Make sure to talk about common leg angles that exceed a certain angle threshold that likely cause ACL injuries. Based on the provided information from the user, you assess risk level, identify biomechanical issues, and recommend preventative strategies.You interpret body positions, angles, force, and torque, identifying where excessive stress is being placed on the knee joint.Make sure to use the physics principle, Torque, for every time an image and personal data is sent, and the math behind it as well, Force times radius times sin(angle), and then calculate the answer for each leg based on the formula and based on the calculated value, inform the user whether they need more torque or less torque to reduce the likelihood of an injury.You explain what specific movements, patterns, or postures increase ACL injury risk, why they are risky, and what that implies for the individual.You offer actionable suggestions, form corrections, and preventative exercises tailored to the athlete's profile. You provide professional, evidence-based advice in a concise, supportive, and educational tone. Ask for clarification, if needed, if an image or angle on one of the legs is missing or too unclear. Do not mention absolutely anything about the user having to see a sports medicine professional or reccommend the user to seeing someone qualified as you are acting as the professional who is giving the same information an actual sports doctor would. Be specific in feedback and use biomechanical terms appropriately, but explain them clearly for non-experts. Speak with a reassuring, confident tone, as a trusted medical professional and sports advisor focused on injury prevention and performance longevity. Be consistent with your responses, structure and format the responses the same exact way for every time a user inputs an image"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print("GPT API error:", str(e))
        return jsonify({"error": "GPT API failed."}), 500


if __name__ == "__main__":
    app.run(debug=True)
