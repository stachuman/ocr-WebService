from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from ocr import ocr_image, summarize_text
from urllib.parse import unquote
import fitz  # PyMuPDF
from PIL import Image
import io
import re
import os
import glob

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/mnt/Public/skany/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OCR function (replace this with your actual OCR engine)
def ocr_engine(image,inst):
    text = ocr_image(image,inst)
    return text

def summarize_engine(text,inst):
    ret_text = summarize_text(text,inst)
    return ret_text

@app.route('/')
def index():
    return render_template('index.html')

def delete_files(directory, pattern):
    """
    Deletes files matching a specific pattern from a directory.

    :param directory: Path to the directory containing the files.
    :param pattern: Pattern to match files (e.g., 'page*.png').
    """
    # Construct the full search path
    search_path = os.path.join(directory, pattern)

    # Find all files matching the pattern
    files_to_delete = glob.glob(search_path)

    # Check if any files were found
    if not files_to_delete:
        print(f"No files found matching the pattern '{pattern}' in '{directory}'.")
        return

    # Delete each file
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Example usage:
# delete_files_from_catalogue('/path/to/catalogue', 'page*.png')


@app.route('/uploadImage', methods=['POST'])
def upload_image():
    #print("111")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    delete_files(app.config['UPLOAD_FOLDER'],'page*.png')

    # Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    #print(filepath)
    
    return jsonify({'filename': file.filename})

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    delete_files(app.config['UPLOAD_FOLDER'],'page*.png')

    # Save the uploaded PDF
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    # Count pages in PDF and return that info
    try:
        pdf_document = fitz.open(filepath)
        page_count = len(pdf_document)
        return jsonify({'filename': file.filename, 'page_count': page_count})
    except Exception as e:
        print(f"Error counting PDF pages: {e}")
        return jsonify({'filename': file.filename})

@app.route('/render_page', methods=['POST'])
def render_page():
    data = request.json
    filename = unquote(data.get('filename'))
    page_number = data.get('page_number', 0) - 1

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    #return(f"Rendering page {page_number} from {filepath}")
    print(f"Rendering page {page_number} from {filepath}", flush=True)
    
    pdf_document = fitz.open(filepath)
    if page_number < 0 or page_number >= len(pdf_document):
        return jsonify({'error': 'Invalid page number'}), 400

    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale (150 DPI)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'page_{page_number}.png')
    img_filename = f'page_{page_number}.png'

    # Convert raw data to an image
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Save to memory with max quality
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG", optimize=True, compress_level=0)
    img_bytes.seek(0)

    # (Optional) Save directly to disk
    with open(img_path, "wb") as f:
        f.write(img_bytes.getvalue())

    #img_bytes = io.BytesIO()
    #img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #img.save(img_bytes, format="PNG")
    #img_bytes.seek(0)

    #with open(img_path, "wb") as f:
    #    f.write(img_bytes.read())

    print(f"Output png: {img_filename}")
    return jsonify({'image_url': url_for('uploaded_file', filename=img_filename, _external=True)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #print(send_from_directory(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ocr_page', methods=['POST'])
def ocr_page():
    data = request.json
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('image_path'))
    instruction = data.get('instruction')

    if not os.path.exists(image_path):
        print(image_path)
        return jsonify({'error': 'Image not found'}), 404

    #img = Image.open(image_path)
    print("Instruction: " + instruction)
    recognized_text = ocr_engine(image_path,instruction) + " "

    return jsonify({'text': recognized_text})

@app.route('/process_all_pages', methods=['POST'])
def process_all_pages():
    data = request.json
    filename = unquote(data.get('filename'))
    instruction = data.get('instruction')
    page_number = data.get('page_number', 0)  # Current page being processed
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        pdf_document = fitz.open(filepath)
        total_pages = len(pdf_document)
        
        # If we've processed all pages, return completion status
        if page_number >= total_pages:
            return jsonify({
                'status': 'complete',
                'message': f'All {total_pages} pages processed',
                'current_page': page_number,
                'total_pages': total_pages
            })
        
        # Process current page
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'page_{page_number}.png')
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG", optimize=True, compress_level=0)
        img_bytes.seek(0)
        
        with open(img_path, "wb") as f:
            f.write(img_bytes.getvalue())
        
        # OCR the current page
        recognized_text = ocr_engine(img_path, instruction) + " "
        
        # Return progress info and text
        return jsonify({
            'status': 'processing',
            'page_text': recognized_text,
            'current_page': page_number,
            'total_pages': total_pages,
            'progress': (page_number + 1) / total_pages * 100
        })
        
    except Exception as e:
        print(f"Error processing PDF page {page_number}: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def clean_text(raw_text):
    """Clean the input text while preserving empty lines and new lines after a sentence."""
    lines = raw_text.splitlines(keepends=True)

    cleaned_lines = []
    buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:  # Keep empty lines
            if buffer:
                cleaned_lines.append(" ".join(buffer))
                buffer = []
            cleaned_lines.append(line)
        elif re.search(r'[.!?…]$', stripped):  # Ends with a sentence-ending punctuation
            buffer.append(stripped)
            cleaned_lines.append(" ".join(buffer) + "\n")  # Keep the new line
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        cleaned_lines.append(" ".join(buffer))  # Add any remaining buffered text

    return "".join(cleaned_lines)  # Join with original newlines


@app.route('/cleanText', methods=['POST'])
def clean_text_endpoint():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400

    cleaned = clean_text(data['text'])
    return jsonify({'cleaned_text': cleaned})

@app.route('/ocr_summary', methods=['POST'])
def ocr_summary():
    data = request.json
    text = data.get('text')
    instruction = data.get('instruction')

    #img = Image.open(image_path)
    print("Instruction: " + instruction)
    summarize = summarize_text(text,instruction)

    return jsonify({'text': summarize})


if __name__ == '__main__':
    #print(app.url_map)

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
