# pip install deepface mediapipe opencv-python pandas

from deepface import DeepFace
import cv2
import mediapipe as mp
import pandas as pd

# 이미지 리스트
image_files = {
    "images2": "images2.jpg",
    "images3": "images3.jpg",
    "images4": "images4.jpg",
    "images5": "images5.jpg",
    "smile1": "smile1.jpg",
    "smile2": "smile2.jpg",
    "smile3": "smile3.jpg",
    "smile4": "smile4.jpg",
    "neutral1": "neutral1.jpg",
    "neutral2": "neutral2.jpg",
    "neutral3": "neutral3.jpg",
}

dict_emotion_kor = {
    "angry": "화남", "disgust": "혐오", "fear": "두려움",
    "happy": "행복", "sad": "슬픔", "surprise": "놀람", "neutral": "무표정"
}

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = []

for name, path in image_files.items():
    result = DeepFace.analyze(img_path=path, actions=['emotion'], detector_backend='mtcnn')
    dominant_emotion = result[0]["dominant_emotion"]

    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mediapipe_result = face_mesh.process(rgb)

    mouth_width = None
    mouth_open_ratio = None
    slope = None
    cheek_diff = None
    lip_asymmetry = None
    judgment = "판단 불가"

    if mediapipe_result.multi_face_landmarks:
        landmarks = mediapipe_result.multi_face_landmarks[0]
        h, w, _ = image.shape

        def get_xy(idx):
            lm = landmarks.landmark[idx]
            return lm.x * w, lm.y * h

        # 입 좌표
        x1, y1 = get_xy(61)   # 왼쪽 입꼬리
        x2, y2 = get_xy(291)  # 오른쪽 입꼬리
        xtop, ytop = get_xy(13)  # 입 위
        xbot, ybot = get_xy(14)  # 입 아래

        # 광대 좌표
        _, cheek_left_y = get_xy(234)
        _, cheek_right_y = get_xy(454)
        cheek_diff = abs(cheek_left_y - cheek_right_y)

        # 코 중심
        xnose, _ = get_xy(2)
        lip_center_x = (x1 + x2) / 2
        center_offset = abs(xnose - lip_center_x)

        # 입 비대칭
        lip_asymmetry = abs(y1 - y2)

        # 입 관련 계산
        mouth_width = abs(x2 - x1)
        mouth_open = abs(ybot - ytop)
        mouth_open_ratio = mouth_open / mouth_width if mouth_width else 0
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

        # ✅ 최종 적합 판단 기준
        if (
            (dominant_emotion in ["neutral", "happy"]) and
            (mouth_open_ratio < 0.12) and
            (mouth_width < 60) and
            (slope <= 0.05) and
            (lip_asymmetry < 4.0) and       # 입꼬리 좌우 높이차
            (cheek_diff < 6.0) and          # 광대 높이 비대칭 허용
            (center_offset < 10.0)          # 입 중앙과 코 수직선의 차
        ):
            judgment = "⭕ 적합"
        else:
            judgment = "❌ 부적합"

    results.append({
        "이미지": name,
        "감정": dict_emotion_kor.get(dominant_emotion, dominant_emotion),
        "입꼬리기울기": round(slope, 4) if slope else "N/A",
        "입꼬리거리(px)": round(mouth_width, 2) if mouth_width else "N/A",
        "입벌어짐비율": round(mouth_open_ratio, 2) if mouth_open_ratio else "N/A",
        "입꼬리비대칭": round(lip_asymmetry, 2) if lip_asymmetry else "N/A",
        "광대비대칭": round(cheek_diff, 2) if cheek_diff else "N/A",
        "입중앙오프셋": round(center_offset, 2) if center_offset else "N/A",
        "최종 판단": judgment
    })

# 결과 출력
import pandas as pd
df = pd.DataFrame(results)
print("\n📊 최종 판단 결과:")
print(df)
