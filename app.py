from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from flask_cors import CORS
 
 # Enable CORS for all routes


app = Flask(__name__)
CORS(app) 
@app.route('/')
def index():
    return render_template('comment.html')

# Load the BERT model and tokenizer for toxicity classification


# Function to classify a comment as toxic or not
def classify_toxicity(input_text, model, tokenizer, device):
    user_input = [input_text]

    user_encodings = tokenizer(
        user_input, truncation=True, padding=True, return_tensors="pt")

    user_dataset = TensorDataset(
        user_encodings['input_ids'], user_encodings['attention_mask'])

    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_probs = torch.sigmoid(logits)

    predicted_label = (predicted_probs.cpu().numpy()[0][0] > 0.5).astype(int)

    # Define class labels for binary classification (0: appropriate, 1: inappropriate)
    class_lab = ['appropriate','inappropriate']
    result = class_lab[predicted_label]
    return result

# Define a route for checking comment toxicity
@app.route('/get_data', methods=['GET'])
@app.route('/check_appropriate', methods=['POST'])
def check_toxicity():
    model_path = 'D:\\Pen Drive\\Comment page'  # Replace with your model path
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    device = torch.device('cpu')
    model = model.to(device)
    try:
        if request.method == 'GET':
            data = request.args.to_dict()
            comment_text = data.get('comment_text','')

        # Classify comment toxicity using the BERT model
        elif request.method == 'POST':
            data = request.get_json()
            comment_text = data.get('comment_text','')

        is_toxic = classify_toxicity(input_text=comment_text,model=model, tokenizer=tokenizer, device=device)
        # Return the result as JSON
        response = {'is_toxic': is_toxic}
        #print(response)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
#check_toxicity(model = model)
if __name__ == '__main__':
    app.run(debug=True)
