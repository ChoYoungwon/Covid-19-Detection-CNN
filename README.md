## 주제 : COVID-19 X-ray 이미지 분류 모델

이 프로젝트는 흉부 X-ray 이미지를 활용하여 COVID-19 폐렴(Pneumonia)을 진단하는 딥러닝 모델을 개발하는 것을 목표로 합니다.  
링크 : https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/data

### 1. 데이터셋

Kagglehub에서 제공하는 'COVID-19 X-ray Dataset'을 사용했습니다. 이 데이터셋은 `train` 및 `test` 세트로 구성되어 있으며, 각 세트는 `NORMAL` (정상) 및 `PNEUMONIA` (폐렴) 클래스로 분류된 X-ray 이미지들을 포함합니다. 폐렴 클래스에는 COVID-19 관련 폐렴 이미지도 포함되어 있습니다.

### 2. 모델 아키텍처

전이 학습(Transfer Learning) 기법을 사용하여 `MobileNetV2` 모델을 기반으로 구축되었습니다.

*   **기본 모델**: ImageNet 데이터셋으로 사전 학습된 `MobileNetV2`를 특징 추출기(Feature Extractor)로 사용했습니다. `include_top=False` 옵션을 통해 MobileNetV2의 최상단 분류 레이어를 제거하고, 기본 모델의 가중치는 초기에는 동결(Freeze)했습니다.
*   **새로운 분류 레이어**: MobileNetV2의 출력에 `GlobalAveragePooling2D` 레이어를 추가하여 특징 맵을 단일 벡터로 변환한 후, `Dense` 레이어(128 뉴런, ReLU 활성화)와 `Dropout` 레이어(0.5)를 추가하여 과적합을 방지했습니다. 최종 출력은 이진 분류를 위한 `Dense` 레이어(1 뉴런, Sigmoid 활성화)로 구성됩니다.

### 3. 데이터 전처리 및 증강

`ImageDataGenerator`를 사용하여 이미지를 전처리하고 증강했습니다.

*   **이미지 크기**: `224x224` 픽셀로 크기를 조정했습니다 (`MobileNetV2` 입력 요구사항).
*   **정규화**: 모든 픽셀 값을 0-1 범위로 스케일링했습니다 (`rescale=1./255`).
*   **데이터 증강 (학습 데이터에만 적용)**: 모델의 일반화 성능 향상을 위해 회전(`rotation_range=20`), 가로/세로 이동(`width_shift_range=0.2`, `height_shift_range=0.2`), 전단 변환(`shear_range=0.2`), 확대/축소(`zoom_range=0.2`), 수평 뒤집기(`horizontal_flip=True`) 등을 적용했습니다.
*   **데이터 분할**: 학습 데이터의 80%는 학습용으로, 20%는 검증용으로 분할했습니다.

### 4. 모델 학습

*   **옵티마이저**: `Adam` 옵티마이저 (`learning_rate=0.0001`)를 사용했습니다.
*   **손실 함수**: 이진 분류 문제이므로 `binary_crossentropy`를 사용했습니다.
*   **평가 지표**: `accuracy` (정확도), `Precision` (정밀도), `Recall` (재현율)을 모니터링했습니다.
*   **콜백**: 학습 과정의 효율성을 높이기 위해 `EarlyStopping` (3 에포크 동안 검증 손실 개선이 없으면 중단)과 `ModelCheckpoint` (최고 성능 모델 저장)를 사용했습니다.

#### 학습 결과

| Epoch | Accuracy | Loss | Precision | Recall | Val_Accuracy | Val_Loss | Val_Precision | Val_Recall |
|-------|----------|------|-----------|--------|--------------|----------|---------------|------------|
| 1     | 0.5333   | 0.7744 | 0.5244    | 0.7167 | 0.6429       | 0.5952   | 0.6111        | 0.7857     |
| ...   | ...      | ...  | ...       | ...    | ...          | ...      | ...           | ...        |
| 9     | 0.8500   | 0.3738 | 0.8750    | 0.8167 | 0.9286       | 0.3125   | 0.8750        | 1.0000     |
| 10    | 0.9083   | 0.3273 | 0.9298    | 0.8833 | 0.9643       | 0.2846   | 0.9333        | 1.0000     |

### 5. 모델 평가

최종 모델은 테스트 데이터셋에서 높은 성능을 보였습니다.

*   **Test Loss**: 0.3115
*   **Test Accuracy**: 0.9750
*   **Test Precision**: 0.9524
*   **Test Recall**: 1.0000

**Classification Report:**
```
              precision    recall  f1-score   support

      NORMAL       0.95      0.90      0.92        20
   PNEUMONIA       0.90      0.95      0.93        20

    accuracy                           0.93        40
   macro avg       0.93      0.93      0.92        40
weighted avg       0.93      0.93      0.92        40
```

**Confusion Matrix:**
```
[[18  2]
 [ 1 19]]
```

### 6. 실행 방법(우분투 환경)  
```
# 파이썬 가상환경에서
pip install -r requirements.txt 
python launch_gradio.py

# gradio
images의 폴더의 Test용 사진 Noraml 2장, SARS-..., strep...로 표기된 covid 이미지로 인터페이스에서 시현해봅니다.
```

### 6. Gradio를 이용한 모델 배포

학습된 모델은 Gradio 라이브러리를 사용하여 간단한 웹 인터페이스로 배포되었습니다. 사용자는 이 인터페이스를 통해 X-ray 이미지를 업로드하고 모델의 예측 결과를 즉시 확인할 수 있습니다. 이를 통해 모델의 실용적인 적용 가능성을 시각적으로 검토할 수 있습니다.

#### 인터페이스 화면
![](../images/display.png)
---