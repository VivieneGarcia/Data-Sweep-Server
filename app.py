from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
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
    
def reformat_date(data, date_format):
    # Define format mappings for supported date formats
    format_mappings = {
        'mm/dd/yyyy': '%m/%d/%Y',   # Correct mapping for month/day/year
        'dd/mm/yyyy': '%d/%m/%Y',   # Correct mapping for day/month/year
        'yyyy/mm/dd': '%Y/%m/%d',   # Correct mapping for year/month/day
    }

    # Check if the date format is supported
    if date_format not in format_mappings:
        raise ValueError(f"Unsupported date format: {date_format}")

    # Get the datetime format string based on the provided date format
    date_format_str = format_mappings[date_format]

    print(f"Target Format: {date_format_str}")

    # Iterate over the rows and columns to reformat the dates
    for row in data:
        for i, value in enumerate(row):
            if isinstance(value, str):
                print(f"Original Value: {value}")
                try:
                    # Try to parse and reformat the date (assuming the date is in '%Y-%m-%d' format)
                    parsed_date = datetime.strptime(value, '%Y-%m-%d')  # Original expected format
                    print(f"Parsed Date: {parsed_date}")
                    
                    reformatted_date = parsed_date.strftime(date_format_str)  # Reformat to the target format
                    print(f"Reformatted Date: {reformatted_date}")

                    row[i] = reformatted_date  # Update the row with the reformatted date
                except ValueError as e:
                    print(f"Skipping value '{value}' due to error: {e}")
                    pass  # Skip if the value can't be parsed as a date

    return data

def apply_date_format(data, columns, date_formats, classifications):
    
    # Define format mappings for supported date formats
    format_mappings = {
        'mm/dd/yyyy': '%m/%d/%Y',  
        'dd/mm/yyyy': '%d/%m/%Y', 
        'yyyy/mm/dd': '%Y/%m/%d',  
    }
    
    for i in range(len(columns)):
        # Check if the column is a date column (based on classification)
        if classifications[i][3] == 1:  # 1 indicates it is a date column
            date_format = date_formats[i]  # Get the desired format for the date column
            
            # Ensure the desired format is valid
            if date_format not in format_mappings:
                raise ValueError(f"Unsupported date format: {date_format}")
            
            # Get the corresponding format string
            format_str = format_mappings[date_format]
            
            # Process each row for the given column
            for j in range(len(data)):
                value = data[j][i]
                
                # Check if the value is a valid string date
                if isinstance(value, str):
                    try:
                        # Try to parse the date assuming the input format is YYYY-MM-DD
                        date_obj = datetime.strptime(value, "%Y-%m-%d")
                        
                        # Format the date according to the specified format
                        data[j][i] = date_obj.strftime(format_str)
                    except ValueError:
                        # If parsing fails, skip this value and leave it unchanged
                        pass

    return data

def count_non_numeric(data, column_index):
    non_numeric_count = 0
    for row in data[1:]:  # Skip header row
        value = row[column_index]
        # Skip None, empty strings, and NaN values
        if value is None or value == " " or pd.isna(value) or value =="":
            continue
        # Check if the value is non-numeric
        try:
            float(value)
        except (ValueError, TypeError):
            non_numeric_count += 1

    return non_numeric_count  # Return the count as an integer

def detect_invalid_dates(data, column_index):
    # Initialize the count of invalid dates
    invalid_dates_count = 0

    # Define valid date formats
    valid_date_formats = [
        '%m-%d-%y',   # MM-DD-YY
        '%d-%m-%y',   # DD-MM-YY
        '%Y-%m-%d',   # YYYY-MM-DD
        '%m/%d/%Y',   # MM-DD-YYYY
        '%d/%m/%Y',   # DD-MM-YYYY
        '%b %d, %Y',  # Jan 01, 2020 (month abbreviation, comma)
        '%b. %d, %Y', # Jan. 01, 2020 (month abbreviation, period, comma)
        '%B %d, %Y',  # February 01, 2004 (full month name, comma)
    ]

    # Iterate through each row in the data, skipping the header row
    for row in data[1:]:  # Assuming data[0] is the header row
        date_value = row[column_index]

        # Skip missing values
        if date_value is None or date_value == "" or date_value == " ":
            continue

        is_valid = False
        for date_format in valid_date_formats:
            try:
                # Try parsing the date with the current format
                datetime.strptime(str(date_value), date_format)
                is_valid = True
                break  # Stop checking other formats if valid
            except ValueError:
                continue  # Try the next date format

        # If the date is invalid, increment the count
        if not is_valid:
            invalid_dates_count += 1

    # Return the count of invalid dates for the specified column
    return invalid_dates_count


def detect_issues(data, columns, classifications):
    issues = {}
    missingValuesCount = 0
    
    for i in range(len(columns)):
        column_issues = []
        column_index = i
        column_data = [row[column_index] for row in data[1:]]  # Skip header row

        # Count missing values in the column
        missing_count = sum(1 for value in column_data if pd.isna(value) or value == " " or value == "")
        missingValuesCount += missing_count
        if missing_count > 0:
            column_issues.append(f"Missing Values")

        if classifications[i][0] == 1:  # Numeric column
            non_numeric_count = count_non_numeric(data, column_index)
            if non_numeric_count > 0:
                column_issues.append(f"Non-Numeric Values")

        
        elif classifications[i][3] == 1:  # Date column
            invalid_dates_count = detect_invalid_dates(data, column_index)
            if invalid_dates_count >= 0:
                column_issues.append(f"Invalid Dates")

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

@app.route('/reformat_invalid_date', methods=['POST'])
def reformat_invalid_dates_route():
    data = request.json['data']
    date_format = request.json['dateFormat'][0]   #[0] Assuming you take the first format if it’s a list
    result = reformat_date(data, date_format)
    return jsonify(result)

if __name__ == '__main__':
    # Set host to 0.0.0.0 to make it accessible on the network, port to 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
