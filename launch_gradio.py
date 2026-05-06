import gradio as gr
import numpy as np
import tensorflow as tf
import cv2

# 이미지 전처리 및 모델 로딩에 필요한 상수 정의
IMG_WIDTH, IMG_HEIGHT = 224, 224
class_names = ['NORMAL', 'PNEUMONIA'] # 모델의 출력 클래스 이름

# 학습된 모델 로드
# best_xray_model.h5 파일이 현재 디렉토리에 있다고 가정합니다.
model = tf.keras.models.load_model('best_xray_model.h5')

# 예측 함수 정의
def predict_image(image):
    # Gradio는 NumPy 배열 형태의 이미지를 전달합니다.
    # 이미지를 모델 입력 크기로 조정하고 정규화합니다.
    if image is None:
        return "Please upload an image."

    img_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0  # 0-1 범위로 정규화
    img_expanded = np.expand_dims(img_normalized, axis=0) # 배치 차원 추가

    # 모델 예측
    prediction = model.predict(img_expanded)[0][0]

    # 예측 결과 해석
    if prediction > 0.5:
        label = class_names[1]  # PNEUMONIA
        confidence = prediction
    else:
        label = class_names[0]  # NORMAL
        confidence = 1 - prediction

    # 결과 포맷팅
    return {
        class_names[0]: float(1 - prediction), # NORMAL 확률
        class_names[1]: float(prediction)      # PNEUMONIA 확률
    }
    
    
# Gradio 인터페이스 생성
# 입력은 이미지, 출력은 라벨 (클래스별 확률)
interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="numpy", label="X-ray Image Upload"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="COVID-19 X-ray Classification",
    description="Upload an X-ray image to classify it as Normal or Pneumonia (including COVID-19)."
)

# 인터페이스 실행
interface.launch()