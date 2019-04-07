

from flask import Flask
from AutismScreeningInDeepLearning import *
from sklearn import model_selection



app = Flask(__name__)

@app.route("/")
def hello():

    test="\n1,0,1,0,1,0,1,0,0,0,4,f,'South Asian',no,no,India,no,3,'4-11 years',Parent,NO"

    file1 = open('Autism-Child-Data.txt',"a") 
    file1.write(test)

    file1.close()
    
    data = data_preprocess()
    prediction= generate_model(data[0],data[1])

    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
