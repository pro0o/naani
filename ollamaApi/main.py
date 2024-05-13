from flask import Flask, request, jsonify
import ollama, json

app = Flask(__name__)
def extract_json_string(response_data):
    start_index = response_data.find("[{")
    end_index = response_data.rfind("}]") + 2
    json_string = response_data[start_index:end_index]
    return json_string

@app.route('/suggestions', methods=['POST'])
def chat():
    if request.is_json:
        content = request.json.get('content', '')
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': content}])

        response_content = response['message']['content']
        json_string = extract_json_string(response_content)
        print(json_string)
        json_data = json.loads(json_string)
        print(json_data)

        return jsonify({'response': json_data})
    else:
        return jsonify({'error': 'Request must be in JSON format'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
