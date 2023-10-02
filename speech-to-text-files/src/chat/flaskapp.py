from flask import Flask, request, jsonify, send_file
import speaker_gpus  # Assuming this is the name of your script

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text_input = data['text']
    speaker_gpus.speak(text_input) 
    return send_file("test.wav", as_attachment=False, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
