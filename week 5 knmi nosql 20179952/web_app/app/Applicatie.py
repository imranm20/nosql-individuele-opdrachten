import numpy as np
from flask import Flask, request, render_template
import pickle

#initialiseren van de app
app = Flask(__name__,template_folder='templates')

#model opvragen die gedumpt is
model = pickle.load(open('modelSAL.pkl', 'rb'))

#default pagina van de applicatie
@app.route('/')
def home():
    return render_template('Voorspellingspage.html')


#waarden uit de form ophalen op het moment de button wordt aangeklikt
@app.route('/v_gemwsnelheid',methods=['POST'])
def voorspellen_wStoot():
    #waarden uit de form ophalen
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    #waarden worden bewaard
    waarden = [np.array(int_features)]
    print(waarden)
    #waarden wordt naar model gestuurd en bewaard in een variabele:
    v = model.predict(waarden)
    print(v)
    
    return render_template('Voorspellingspage.html', antwoord = 'De voorspelde hardste windstoot is: ' + v )

if __name__ == "__main__":
    app.run(debug=False)