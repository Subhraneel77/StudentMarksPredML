from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"]) # when we do the POST, the request.form will get the entire information

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data=CustomData(
            gender=request.form.get('gender'),  #to get the information aout gender
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
    new_data = data.get_data_as_data_frame() #to convert the entire CustomData present above to data frame.To do this I am calling the get_data_as_data_frame() function.
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data) #it will go to prediction_pipleine and will give the results.

    results = round(pred[0],2) 

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(port=8000, debug= True)

#http://127.0.0.1:8000/ in browser