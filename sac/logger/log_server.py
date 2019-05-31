from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='.')


def get_content_from(filename, errmsg='No logs found.'):
    try:
        with open(filename, 'r+') as file:
            content = file.read()
    except FileNotFoundError:
        content = errmsg
    return content


@app.route('/')
def log():
    content = get_content_from('logs.txt')
    return render_template('content.html', text=content)


@app.route('/<int:episode>')
def actions(episode):
    path = os.path.abspath(f'logger/actions/{episode}.txt')
    parent = os.path.abspath(os.path.join(path, os.pardir))
    actions_path = os.path.abspath('logger/actions')
    print('actions path:', actions_path)
    print('parent path:', parent)
    if not actions_path == parent:
        content = 'Please don\'t hack me'
    else:
        content = get_content_from(f'actions/{episode}.txt', f'Not passed episode {episode} yet.')
    return render_template('content.html', text=content)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
