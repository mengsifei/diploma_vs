from flask import Flask, request, render_template
import torch
import torch.nn as nn
from models.electra_baseline import *
from utils.test import *
from transformers import ElectraTokenizer
app = Flask(__name__)

# Load the PyTorch model
model = BaseModel()
model.load_state_dict(torch.load('checkpoints/best_model_electra_simple_multiscale_4_8_12.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input1 = request.form['input1']
        input2 = request.form['input2']
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        scores = score_essay_vanilla(input1, input2, tokenizer, model, 'cpu')
        return render_template('index.html', result=scores)
    else:
        return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
