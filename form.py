from flask import Flask, request, jsonify

app = Flask(__name__)

# Define route for serving the HTML page
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Topic Modeling Demo</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <h1>Topic Modeling Demo</h1>
        <form id="topicForm">
            <label for="textInput">Enter text:</label><br>
            <input type="text" id="textInput" name="text"><br>
            <button type="submit">Submit</button>
        </form>
        <div id="output"></div>
        <script>
            $(document).ready(function() {
                $('#topicForm').submit(function(event) {
                    event.preventDefault();
                    var text = $('#textInput').val();
                    $.ajax({
                        url: '/topic_modeling',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ text: text }),
                        success: function(response) {
                            $('#output').text(JSON.stringify(response));
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """

# Define route for topic modeling
@app.route('/topic_modeling', methods=['POST'])
def topic_modeling():
    # Your topic modeling code here
    return jsonify({'message': 'Topic modeling result will be displayed here'})

if __name__ == '__main__':
    app.run(debug=True)