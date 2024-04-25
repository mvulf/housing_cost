from flask import Flask, request, jsonify
import pickle
import pandas as pd

exp10 = lambda x: 10 ** x 

def exp_predict(
    model, 
    X:pd.DataFrame
):
    """ Make model prediction and exponentiate it

    Args:
        model: prediction model
        X: dataset for prediction

    Returns:
        real prediction
    """
    log_prediction = model.predict(X)
    prediction = exp10(log_prediction)
    
    return prediction

# Import model from the file
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Server is running"
    return msg

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data = pd.DataFrame(data)
    # data.info()
    # print(data)
    
    prediction = list(
        exp_predict(
            model=model,
            X=data
        )
    )
    
    return jsonify(
        {
            'prediction': prediction
        }
    )

if __name__ == '__main__':
    app.run(host='localhost', port=4000)