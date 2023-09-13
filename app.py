from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData , PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            pclass = int(request.form['pclass']),
            sex = request.form['sex'],
            age = float(request.form['age']),
            sibsp = int(request.form['sibsp']),
            parch = int(request.form['parch']),
            fare = float(request.form['fare']),
            cabin = request.form['cabin'],
            embarked = request.form['embarked'],
            title = request.form['title']
            )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        if pred[0] == 0.0:
            result = 'Death'
        else:
            result = 'Survived'

        
        return render_template('form.html',final_result = result)
    

if __name__ == '__main__':
    app.run(debug=True)
