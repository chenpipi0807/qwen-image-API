from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import json
import os
import base64
import time
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from PIL import Image, ImageOps
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load API keys
def load_api_keys():
    try:
        with open('api-key.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

API_KEYS = load_api_keys()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def expand_image_to_ratio(image_path, target_ratio, max_dimension):
    """Expand image to target aspect ratio with white padding and resize to max dimension"""
    # Parse target ratio
    ratio_parts = target_ratio.split(':')
    target_width_ratio = float(ratio_parts[0])
    target_height_ratio = float(ratio_parts[1])
    target_aspect = target_width_ratio / target_height_ratio
    
    # Open and process image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size
        original_aspect = original_width / original_height
        
        # Calculate new dimensions for expansion
        if original_aspect > target_aspect:
            # Image is wider than target ratio, add padding to height
            new_width = original_width
            new_height = int(original_width / target_aspect)
        else:
            # Image is taller than target ratio, add padding to width
            new_height = original_height
            new_width = int(original_height * target_aspect)
        
        # Create new image with white background
        expanded_img = Image.new('RGB', (new_width, new_height), 'white')
        
        # Calculate position to paste original image (center it)
        paste_x = (new_width - original_width) // 2
        paste_y = (new_height - original_height) // 2
        
        # Paste original image onto white background
        expanded_img.paste(img, (paste_x, paste_y))
        
        # Resize to max dimension while maintaining aspect ratio
        expanded_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # Save expanded image
        expanded_path = image_path.replace('.', '_expanded.')
        expanded_img.save(expanded_path, 'JPEG', quality=95)
        
        return expanded_path

def encode_image_to_base64(file_path):
    """Convert image file to base64 encoding"""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Get file extension to determine MIME type
        ext = file_path.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        return f"data:{mime_type};base64,{encoded_string}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_page():
    return render_template('generate.html')

@app.route('/edit')
def edit_page():
    return render_template('edit.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate image using Qwen Image Generation API"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': '请输入图像描述'}), 400
        
        # API request headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEYS.get("qwen-api-key", "")}'
        }
        
        # Get additional parameters
        negative_prompt = data.get('negative_prompt', '')
        size = data.get('size', '1328*1328')
        prompt_extend = data.get('prompt_extend', True)
        watermark = data.get('watermark', False)
        
        # API request payload
        payload = {
            "model": "qwen-image",
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "size": size,
                "n": 1,
                "prompt_extend": prompt_extend,
                "watermark": watermark
            }
        }
        
        # Add negative prompt if provided
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        
        # Add async header for proper API call
        headers['X-DashScope-Async'] = 'enable'
        
        # Step 1: Create task
        response = requests.post(
            'https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return jsonify({'error': f'API请求失败: {response.text}'}), 500
        
        result = response.json()
        
        if result.get('output', {}).get('task_status') == 'SUCCEEDED':
            # Synchronous response
            image_url = result['output']['results'][0]['url']
            return jsonify({
                'success': True,
                'image_url': image_url,
                'task_id': result.get('output', {}).get('task_id')
            })
        elif result.get('output', {}).get('task_id'):
            # Asynchronous response - need to poll for results
            task_id = result['output']['task_id']
            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': 'processing'
            })
        else:
            return jsonify({'error': '图像生成失败'}), 500
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/check-task/<task_id>')
def check_task(task_id):
    """Check the status of an image generation task"""
    try:
        headers = {
            'Authorization': f'Bearer {API_KEYS.get("qwen-api-key", "")}'
        }
        
        response = requests.get(
            f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}',
            headers=headers
        )
        
        if response.status_code != 200:
            return jsonify({'error': '任务查询失败'}), 500
        
        result = response.json()
        
        if result.get('output', {}).get('task_status') == 'SUCCEEDED':
            image_url = result['output']['results'][0]['url']
            return jsonify({
                'success': True,
                'status': 'completed',
                'image_url': image_url
            })
        elif result.get('output', {}).get('task_status') == 'FAILED':
            return jsonify({
                'success': False,
                'status': 'failed',
                'error': result.get('output', {}).get('message', '任务失败')
            })
        else:
            return jsonify({
                'success': True,
                'status': 'processing'
            })
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/edit-image', methods=['POST'])
def edit_image():
    """Edit image using Qwen Image Edit API"""
    try:
        # Check if image file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': '请上传图像文件'}), 400
        
        file = request.files['image']
        edit_prompt = request.form.get('edit_prompt', '')
        enable_expansion = request.form.get('enable_expansion') == 'true'
        target_ratio = request.form.get('target_ratio', '1:1')
        max_dimension = int(request.form.get('max_dimension', 1536))
        
        if file.filename == '':
            return jsonify({'error': '请选择图像文件'}), 400
        
        if not edit_prompt:
            return jsonify({'error': '请输入编辑指令'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            try:
                # Apply image expansion if enabled
                if enable_expansion:
                    file_path = expand_image_to_ratio(file_path, target_ratio, max_dimension)
                
                # Convert image to base64
                image_base64 = encode_image_to_base64(file_path)
                
                # API request headers
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {API_KEYS.get("qwen-api-key", "")}'
                }
                
                # API request payload
                payload = {
                    "model": "qwen-image-edit",
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "image": image_base64
                                    },
                                    {
                                        "text": edit_prompt
                                    }
                                ]
                            }
                        ]
                    },
                    "parameters": {
                        "negative_prompt": "",
                        "watermark": False
                    }
                }
                
                response = requests.post(
                    'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
                    headers=headers,
                    json=payload
                )
                
                # Clean up uploaded file
                os.remove(file_path)
                
                if response.status_code != 200:
                    return jsonify({'error': f'API请求失败: {response.text}'}), 500
                
                result = response.json()
                
                if result.get('output', {}).get('choices'):
                    # Extract image URL from response
                    choice = result['output']['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        content = choice['message']['content']
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'image' in item:
                                    return jsonify({
                                        'success': True,
                                        'image_url': item['image']
                                    })
                
                return jsonify({'error': '图像编辑失败，未找到结果图像'}), 500
                
            except Exception as e:
                # Clean up uploaded file in case of error
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
        else:
            return jsonify({'error': '不支持的文件格式'}), 400
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
