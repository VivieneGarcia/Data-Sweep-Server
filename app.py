from flask import Flask, request, jsonify, send_file
import pandas as pd
import io
from datetime import datetime
import traceback

app = Flask(__name__)

def to_title_case(string):
    return ' '.join([word.capitalize() if len(word) > 1 else word.upper() for word in string.split(' ')])

def to_sentence_case(string):
    if not string:
        return string
    return string[0].upper() + string[1:].lower()

def apply_letter_casing(data, columns, casing_selections):
    for i in range(len(columns)):
        casing = casing_selections[i]
        for j in range(len(data)):
            value = data[j][i]
            if isinstance(value, str):
                if casing == 'UPPERCASE':
                    data[j][i] = value.upper()
                elif casing == 'lowercase':
                    data[j][i] = value.lower()
                elif casing == 'Title Case':
                    data[j][i] = to_title_case(value)
                elif casing == 'Sentence case':
                    data[j][i] = to_sentence_case(value)
    
    return data

def is_valid_date(value):
    try:
        parsed_date = datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def apply_date_format(data, columns, date_formats, classifications):
    for i in range(len(columns)):
        if classifications[i][3] == 1:  # Date column
            date_format = date_formats[i]
            for j in range(len(data)):
                value = data[j][i]
                if value and isinstance(value, str):
                    try:
                        date = datetime.strptime(value, "%Y-%m-%d")
                        data[j][i] = date.strftime(date_format)
                    except ValueError:
                        pass
    return data

def detect_issues(data, columns, classifications):
    issues = {}
    for i in range(len(columns)):
        column_issues = []
        column_data = [row[i] for row in data[1:]]
        
        # Check for missing values
        if any(value is None or value == "" for value in column_data):
            column_issues.append("Missing values")

        # Numeric validation
        if classifications[i][0] == 1:
            if any(value is not None and not isinstance(value, (int, float)) for value in column_data):
                column_issues.append("Contains non-numeric values")

        # Date validation
        elif classifications[i][3] == 1:
            if any(not is_valid_date(value) for value in column_data):
                column_issues.append("Invalid date format")

        if column_issues:
            issues[columns[i]] = column_issues

    return issues

@app.route('/apply_letter_casing', methods=['POST'])
def apply_letter_casing_route():
    data = request.json['data']
    columns = request.json['columns']
    casing_selections = request.json['casingSelections']
    result = apply_letter_casing(data, columns, casing_selections)
    return jsonify(result)

@app.route('/apply_date_format', methods=['POST'])
def apply_date_format_route():
    data = request.json['data']
    columns = request.json['columns']
    date_formats = request.json['dateFormats']
    classifications = request.json['classifications']
    result = apply_date_format(data, columns, date_formats, classifications)
    return jsonify(result)

@app.route('/detect_issues', methods=['POST'])
def detect_issues_route():
    data = request.json['data']
    columns = request.json['columns']
    classifications = request.json['classifications']
    result = detect_issues(data, columns, classifications)
    return jsonify(result)


@app.route('/remove_columns', methods=['POST'])
def remove_columns():
    try:
        data = request.json.get('data')
        columns = request.json.get('columns')
        columns_to_remove = request.json.get('columnsToRemove', [])

        if not data or not columns:
            return jsonify({'error': 'No data or columns provided'}), 400
        if not columns_to_remove:
            return jsonify({'error': 'No columns specified to remove'}), 400

        # Convert the JSON data to a DataFrame
        df = pd.DataFrame(data[1:], columns=columns)

        # Drop duplicates
        df = df.drop_duplicates()

        # Remove the specified columns
        df_cleaned = df.drop(columns=columns_to_remove)

        # Convert the cleaned DataFrame back to JSON
        result_data = [df_cleaned.columns.tolist()] + df_cleaned.values.tolist()
        return jsonify(result_data)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Set host to 0.0.0.0 to make it accessible on the network, port to 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
