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
    print("="*50)
    print("图像编辑请求开始")
    print(f"请求方法: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"表单数据: {dict(request.form)}")
    print(f"文件数据: {list(request.files.keys())}")
    print("="*50)
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
                # Check image dimensions before processing
                with Image.open(file_path) as img:
                    original_width, original_height = img.size
                    file_size = os.path.getsize(file_path)
                    print(f"原始图像信息:")
                    print(f"  - 尺寸: {original_width}x{original_height}")
                    print(f"  - 文件大小: {file_size / (1024*1024):.2f} MB")
                    print(f"  - 格式: {img.format}")
                    print(f"  - 模式: {img.mode}")
                    
                    # Check API requirements
                    if original_width < 384 or original_height < 384:
                        print(f"警告: 图像尺寸过小 ({original_width}x{original_height})，API要求最小384像素")
                    if original_width > 3072 or original_height > 3072:
                        print(f"警告: 图像尺寸过大 ({original_width}x{original_height})，API要求最大3072像素")
                    if file_size > 10 * 1024 * 1024:
                        print(f"警告: 文件大小过大 ({file_size / (1024*1024):.2f} MB)，API要求最大10MB")
                
                # Apply image expansion if enabled
                if enable_expansion:
                    print(f"启用智能扩图: {target_ratio}, 最大尺寸: {max_dimension}")
                    file_path = expand_image_to_ratio(file_path, target_ratio, max_dimension)
                    
                    # Check expanded image dimensions
                    with Image.open(file_path) as expanded_img:
                        exp_width, exp_height = expanded_img.size
                        exp_file_size = os.path.getsize(file_path)
                        print(f"扩图后图像信息:")
                        print(f"  - 尺寸: {exp_width}x{exp_height}")
                        print(f"  - 文件大小: {exp_file_size / (1024*1024):.2f} MB")
                
                # Convert image to base64
                print("开始转换图像为Base64...")
                image_base64 = encode_image_to_base64(file_path)
                print(f"Base64编码完成，长度: {len(image_base64)} 字符")
                
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
                
                print(f"=== 发送API请求 ===")
                print(f"URL: https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
                print(f"编辑指令: {edit_prompt}")
                print(f"API Key: {API_KEYS.get('qwen-api-key', 'NOT_FOUND')[:20]}...")
                print(f"Base64图像长度: {len(image_base64)} 字符")
                print(f"请求参数: enable_expansion={enable_expansion}, target_ratio={target_ratio}, max_dimension={max_dimension}")
                
                start_time = time.time()
                print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
                
                try:
                    response = requests.post(
                        'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"=== API响应 ===")
                    print(f"响应时间: {duration:.2f}秒")
                    print(f"状态码: {response.status_code}")
                    print(f"响应头: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        print(f"响应内容: {response.text[:1000]}...")
                    else:
                        print(f"错误响应: {response.text}")
                        
                except requests.exceptions.Timeout:
                    print("请求超时 (60秒)")
                    raise Exception("API请求超时")
                except requests.exceptions.RequestException as e:
                    print(f"请求异常: {str(e)}")
                    raise
                
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
