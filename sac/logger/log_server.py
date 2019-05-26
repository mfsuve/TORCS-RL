from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')


@app.route('/')
def log():
    try:
        with open('logs.txt', 'r+') as text:
            content = text.read()
    except FileNotFoundError:
        content = 'No logs found.'
    return render_template('content.html', text=content)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
