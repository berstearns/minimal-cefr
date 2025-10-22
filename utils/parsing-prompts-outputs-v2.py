import re
import json
import os
from pathlib import Path
import argparse

def parse_prompts_outputs(prompt_output):
    '''
    Given a prompt output that contains a JSON-like structure (e.g. '{"score":"A1", "level":"B2"}'),
    this function will extract the entire dictionary as a string, or None if no dictionary is found.

    Handles malformed JSON cases like:
    - Missing closing brace: {"score": "B1"
    - Extra characters: {"score": "B1"°}
    - Text before JSON: Here is the JSON file: {"score": "B1"

    Args:
    prompt_output (str): The string containing the dictionary-like pattern.

    Returns:
    dict: A dictionary of the captured JSON object (if valid), or None if no valid dictionary found.
    '''
    if not prompt_output or not isinstance(prompt_output, str):
        return None

    # Strategy 1: Try to find a complete, well-formed JSON object
    match = re.search(r'\{[^{}]*\}', prompt_output)
    if match:
        dict_str = match.group(0)
        # Try to parse it directly with json.loads
        result = try_parse_json(dict_str)
        if result is not None:
            return result

    # Strategy 2: Look for common patterns and try to fix them
    # Find {"score": "XX" or {"level": "XX" patterns
    patterns = [
        r'\{\s*"score"\s*:\s*"([A-C][12])"\s*[°]?\}?',  # Handles {"score": "B1"} or {"score": "B1"°} or {"score": "B1"
        r'\{\s*"level"\s*:\s*"([A-C][12])"\s*[°]?\}?',  # Handles {"level": "B1"} variants
        r'\{\s*"label"\s*:\s*"([A-C][12])"\s*[°]?\}?',  # Handles {"label": "B1"} variants
        r'\{\s*"cefr"\s*:\s*"([A-C][12])"\s*[°]?\}?',   # Handles {"cefr": "B1"} variants
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt_output, re.IGNORECASE)
        if match:
            cefr_level = match.group(1).upper()
            # Determine which key was used
            if 'score' in pattern:
                key = 'score'
            elif 'level' in pattern:
                key = 'level'
            elif 'label' in pattern:
                key = 'label'
            else:
                key = 'cefr'
            return {key: cefr_level}

    # Strategy 3: Try to find any {"key": "value"} pattern and clean it up
    # This handles cases like: {"score": "B1" (missing closing brace)
    match = re.search(r'\{\s*"([^"]+)"\s*:\s*"([^"]+)"', prompt_output)
    if match:
        key = match.group(1)
        value = match.group(2).strip()
        # Validate if it's a CEFR level
        if re.match(r'^[A-C][12]$', value, re.IGNORECASE):
            return {key: value.upper()}
        # Return the key-value pair even if not a CEFR level
        return {key: value}

    # Strategy 4: Try to extract just the CEFR level if all else fails
    # Look for patterns like: A1, A2, B1, B2, C1, C2
    match = re.search(r'\b([A-C][12])\b', prompt_output)
    if match:
        cefr_level = match.group(1).upper()
        return {"score": cefr_level}

    return None


def try_parse_json(json_str):
    '''
    Attempt to parse a JSON string with multiple strategies.

    Args:
    json_str (str): String to parse as JSON

    Returns:
    dict: Parsed dictionary or None if parsing fails
    '''
    # Clean up common issues
    json_str = json_str.strip()

    # Remove trailing characters like ° that might appear before closing brace
    json_str = re.sub(r'[°\s]*\}', '}', json_str)

    # Try direct JSON parsing first (safest method)
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try adding missing closing brace
    if json_str.count('{') > json_str.count('}'):
        try:
            return json.loads(json_str + '}')
        except (json.JSONDecodeError, ValueError):
            pass

    # Try eval as last resort (less safe but handles some edge cases)
    try:
        result = eval(json_str)
        if isinstance(result, dict):
            return result
    except:
        pass

    return None


def convert_to_one_hot_probability(parsed_dict, label_key='score'):
    '''
    Convert a parsed dictionary with a label to one-hot probability format.

    Args:
    parsed_dict (dict): Dictionary containing the label (e.g., {"score": "B1", "level": "B2"})
    label_key (str): Key to extract the CEFR level from (default: 'score')

    Returns:
    dict: One-hot encoded probability dictionary or None if label not found
    '''
    # Define all CEFR levels
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    # Try to extract the label from the parsed dictionary
    label = None

    # First try the specified label_key
    if label_key in parsed_dict:
        label = parsed_dict[label_key]
    # Try common alternative keys
    elif 'level' in parsed_dict:
        label = parsed_dict['level']
    elif 'label' in parsed_dict:
        label = parsed_dict['label']
    elif 'cefr' in parsed_dict:
        label = parsed_dict['cefr']
    # If dict has only one key-value pair, use its value
    elif len(parsed_dict) == 1:
        label = list(parsed_dict.values())[0]

    # Validate the label is a string and is a valid CEFR level
    if label and isinstance(label, str):
        label = label.upper().strip()
        if label in cefr_levels:
            # Create one-hot encoding
            one_hot = {level: 0.0 for level in cefr_levels}
            one_hot[label] = 1.0
            return one_hot

    return None


def process_prompt_files(input_folder, output_folder, use_one_hot=True):
    '''
    Process all .txt files from input_folder, parse their outputs,
    and save valid JSON results to output_folder.

    Args:
    input_folder (str): Path to folder containing .txt prompt output files
    output_folder (str): Path to folder where parsed JSON files will be saved
    use_one_hot (bool): If True, convert labels to one-hot probability format (default: True)

    Returns:
    dict: Statistics dictionary with counts of successful and failed parses
    '''
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize statistics
    stats = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'failed_files': []
    }

    # Get all .txt files from input folder
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return stats

    txt_files = list(input_path.glob('*.txt'))
    stats['total_files'] = len(txt_files)

    print(f"Found {stats['total_files']} .txt files to process")
    print(f"Processing files from: {input_folder}")
    print(f"Saving results to: {output_folder}\n")

    # Process each file
    for txt_file in txt_files:
        try:
            # Read the file content
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the content
            parsed_result = parse_prompts_outputs(content)

            if parsed_result is not None:
                # Convert to one-hot probability format if requested
                if use_one_hot:
                    one_hot_result = convert_to_one_hot_probability(parsed_result)
                    if one_hot_result is not None:
                        final_result = one_hot_result
                    else:
                        # If conversion fails, keep original format
                        final_result = parsed_result
                        print(f"⚠ Warning: Could not convert {txt_file.name} to one-hot format, saving original")
                else:
                    final_result = parsed_result

                # Save as JSON file with same name but .json extension
                output_file = output_path / f"{txt_file.stem}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)

                stats['successful'] += 1
                print(f"✓ Processed: {txt_file.name} -> {output_file.name}")
            else:
                stats['failed'] += 1
                stats['failed_files'].append(txt_file.name)
                print(f"✗ Failed to parse: {txt_file.name}")

        except Exception as e:
            stats['failed'] += 1
            stats['failed_files'].append(txt_file.name)
            print(f"✗ Error processing {txt_file.name}: {str(e)}")

    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Successfully parsed:   {stats['successful']} ({stats['successful']/stats['total_files']*100:.1f}%)" if stats['total_files'] > 0 else "Successfully parsed:   0")
    print(f"Failed to parse:       {stats['failed']} ({stats['failed']/stats['total_files']*100:.1f}%)" if stats['total_files'] > 0 else "Failed to parse:       0")

    if stats['failed_files']:
        print(f"\nFailed files:")
        for failed_file in stats['failed_files']:
            print(f"  - {failed_file}")

    print("="*60)

    return stats


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Parse prompt output .txt files and convert valid ones to JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process files from input_prompts/ and save to output_json/
  python parsing-prompts-outputs-v2.py --input input_prompts --output output_json

  # Run unit tests
  python parsing-prompts-outputs-v2.py --test
        '''
    )
    parser.add_argument('--input', '-i', type=str,
                        help='Input folder containing .txt prompt output files')
    parser.add_argument('--output', '-o', type=str,
                        help='Output folder for parsed JSON files')
    parser.add_argument('--raw-labels', action='store_true',
                        help='Save raw label format instead of one-hot probability format (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests on parse_prompts_outputs function')

    args = parser.parse_args()

    # Run tests if --test flag is provided
    if args.test:
        print("Running unit tests...\n")
        # Test cases including the new failing scenarios
        examples = [
            # Original test cases
            ('{"score":"A1", "level":"B2"} extra text here', {"score": "A1", "level": "B2"}),
            ('{"score":"B2"} and some more data', {"score": "B2"}),
            ('text before {"score":"C1", "level":"A2"} text after', {"score": "C1", "level": "A2"}),
            ('random text {"level":"B1"}', {"level": "B1"}),
            ('Here is a score {"score":"C1"} with some random text.', {"score": "C1"}),

            # NEW: Failing scenarios from user
            ('Here is the JSON file:\n\n{"score": "B1"', {"score": "B1"}),  # Missing closing brace with text before
            ('{"score": "B1"', {"score": "B1"}),  # Missing closing brace
            ('{"score": "B1"°}', {"score": "B1"}),  # Extra ° character

            # Additional edge cases
            ('{"score": "B1" ', {"score": "B1"}),  # Missing brace with trailing space
            ('The level is {"level": "C2"', {"level": "C2"}),  # Text before, missing brace
            ('{"score": "A2"°', {"score": "A2"}),  # Missing brace AND ° character
            ('Result: {"cefr": "B2"°}', {"cefr": "B2"}),  # With ° and text before

            # Should still fail gracefully
            ('no json here at all', None),
            ('{"invalid": "X9"}', {"invalid": "X9"}),  # Not a CEFR level but should parse
        ]

        # Test all examples
        passed = 0
        failed = 0
        for idx, (output, expected) in enumerate(examples):
            result = parse_prompts_outputs(output)
            status = "✓ PASS" if result == expected else "✗ FAIL"
            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"Example {idx+1}: {status}")
            print(f"  Input:    {repr(output)}")
            print(f"  Expected: {expected}")
            print(f"  Result:   {result}")
            if result != expected:
                print(f"  >>> MISMATCH!")
            print()

        print("="*60)
        print(f"Test Results: {passed} passed, {failed} failed out of {len(examples)} total")
        print("="*60)

    # Process files if both input and output folders are provided
    elif args.input and args.output:
        # Use one-hot encoding by default, unless --raw-labels is specified
        use_one_hot = not args.raw_labels
        stats = process_prompt_files(args.input, args.output, use_one_hot=use_one_hot)

    # Show usage if no arguments provided
    else:
        parser.print_help()
