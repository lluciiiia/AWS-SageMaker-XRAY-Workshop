from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import io
from flask_cors import CORS
import sys
import traceback
from vit_keras import vit  # Thư viện Vision Transformer
import threading  # Thêm thư viện để xử lý đa luồng



app = Flask(__name__)
CORS(app)

# Định nghĩa lớp tùy chỉnh ClassToken
class ClassToken(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, 1, input_shape[-1]), initializer="zeros", trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token = tf.broadcast_to(self.w, [batch_size, 1, tf.shape(inputs)[-1]])
        return tf.concat([class_token, inputs], axis=1)

# Định nghĩa lớp tùy chỉnh AddPositionEmbs
class AddPositionEmbs(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddPositionEmbs, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pe = self.add_weight(
            "pe",
            shape=[1, input_shape[1], input_shape[2]],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.pe

# Định nghĩa lớp tùy chỉnh TransformerBlock
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define your models and their respective paths
models = {
    'Vision Transformer': {'path': './VisionTransformer.h5', 'input_size': (224, 224, 3)},
    'DenseNet121': {'path': './DenseNet3.h5', 'input_size': (500, 500, 3)},
    'DenseNet169': {'path': './DenseNet169_model.h5', 'input_size': (224, 224, 3)},
    'EfficientNetV2': {'path': './EfficientNetV2.h5', 'input_size': (224, 224, 3)},
    #'CNN': {'path': './sequential_model.h5', 'input_size': (500, 500, 1)},
    #'ResNet50': {'path': './resnet50_custom.h5', 'input_size': (500, 500, 3)},
    #'VGG16': {'path': './vgg16_1.h5', 'input_size': (500, 500, 3)},
    #'ResNet101': {'path': './ResNet101_model.h5', 'input_size': (224, 224, 3)},
    #'EfficientNetB3': {'path': './EfficientNetB3_model.h5', 'input_size': (300, 300, 3)},
}

# Load model function
def load_model(model_name):
    custom_objects = {
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs,
        "TransformerBlock": TransformerBlock,
        "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
    }
    try:
        with tf.keras.utils.custom_object_scope({
            "ClassToken": ClassToken,
            "AddPositionEmbs": AddPositionEmbs,
            "TransformerBlock": TransformerBlock,
            "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
        }):
            model = tf.keras.models.load_model(models[model_name]['path'])
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        traceback.print_exc(file=sys.stdout)  # In chi tiết lỗi
        return None




# Load all models into a dictionary
loaded_models = {name: load_model(name) for name in models.keys()}

# Preprocess image function
def preprocess_image(image, input_size):
    try:
        # Kiểm tra xem image có phải là đối tượng PIL Image không
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")

        # In thông tin debug
        print(f"Original image mode: {image.mode}")
        print(f"Original image size: {image.size}")
        print(f"Target input size: {input_size}")

        processed_image = None
        if input_size[2] == 1:
            # Xử lý ảnh trắng đen cho mô hình CNN
            processed_image = image.convert('L')  # Chuyển đổi hình ảnh sang grayscale
            processed_image = processed_image.resize(input_size[:2])  # Chỉ lấy (height, width)
            processed_image = np.array(processed_image)
            processed_image = np.expand_dims(processed_image, axis=-1)  # Thêm một trục cho kênh
        else:
            # Xử lý ảnh RGB cho các mô hình khác
            processed_image = image.convert('RGB')
            processed_image = processed_image.resize(input_size[:2])  # Chỉ lấy (height, width)
            processed_image = np.array(processed_image)

        processed_image = processed_image.astype('float32') / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)

        # In thông tin debug
        print(f"Processed image shape: {processed_image.shape}")
        return processed_image

    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        raise


# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided!"}), 400

        image_file = request.files['image']
        
        if not image_file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Kiểm tra định dạng file
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not '.' in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file format"}), 400
        
        # Log thông tin về file
        print(f"Processing file: {image_file.filename}")
        
        # Đọc ảnh
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Log thông tin về ảnh
        print(f"Image mode: {image.mode}")
        print(f"Image size: {image.size}")
        
        results = {}
        for model_name, model_info in models.items():
            try:
                model = loaded_models[model_name]
                if model is None:
                    results[model_name] = "Model failed to load"
                    continue
                
                input_size = model_info['input_size']
                processed_image = preprocess_image(image, input_size)
                
                prediction = model.predict(processed_image)
                result = 'Có bệnh' if prediction[0][0] > 0.5 else 'Bình thường'
                results[model_name] = result
                
                print(f"Prediction for {model_name}: {result}")
                
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
                results[model_name] = f"Error: {str(e)}"

        return jsonify(results)
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)