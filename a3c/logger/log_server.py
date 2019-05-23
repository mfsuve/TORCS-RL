from flask import Flask, request, render_template
import time

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['POST'])
def index():
    msg = request.form['msg']
    with open('logs.txt', 'a+') as log_file:
        time_str = time.strftime('%d-%b-%y %H:%M:%S', time.localtime())
        print(f'{time_str}\t{msg}', file=log_file)
        print(f'{time_str}\t{msg}')
    return ""

@app.route('/log')
def log():
    try:
        with open('logs.txt', 'r+') as text:
            content = text.read()
    except FileNotFoundError:
        content = 'No logs found.'
    return render_template('content.html', text=content)

@app.route('/log/<int:process>', methods=['GET'])
def log_worker(process):
    try:
        with open(f'../../../logs/Worker_{process}.txt', 'r+') as text:
            content = text.read()
    except FileNotFoundError as e:
        content = f'Process {process} haven\'t logged anything.'
    return render_template('content.html', text=content)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3000, debug=True)
