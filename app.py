from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/temperature_info')
def temperature_info():
    return render_template('temperature_info.html')

@app.route('/dissolved_oxygen_info')
def dissolved_oxygen_info():
    return render_template('dissolved_oxygen_info.html')

if __name__ == '__main__':
    app.run(debug=True)
