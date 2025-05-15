def analyze_face(image_path):
    from deepface import DeepFace
    import cv2
    import mediapipe as mp

    dict_emotion_kor = {
        "angry": "화남", "disgust": "혐오", "fear": "두려움",
        "happy": "행복", "sad": "슬픔", "surprise": "놀람", "neutral": "무표정"
    }

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='mtcnn')
        dominant_emotion = result[0]["dominant_emotion"]
    except Exception as e:
        return {"error": f"DeepFace 분석 실패: {str(e)}"}

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mediapipe_result = face_mesh.process(rgb)

    # 초기화
    mouth_width = None
    mouth_open_ratio = None
    slope = None
    cheek_diff = None
    lip_asymmetry = None
    eyebrow_hidden = "판단불가"
    ear_hidden = "판단불가"
    gaze_result = "판단불가"
    face_straight = "판단불가"
    center_offset = None
    judgment = "판단 불가"

    if mediapipe_result.multi_face_landmarks:
        landmarks = mediapipe_result.multi_face_landmarks[0]
        h, w, _ = image.shape

        def get_xy(idx):
            lm = landmarks.landmark[idx]
            return lm.x * w, lm.y * h

        def get_x(idx):
            return landmarks.landmark[idx].x * w

        def get_y(idx):
            return landmarks.landmark[idx].y * h

        # 입 관련
        x1, y1 = get_xy(61)
        x2, y2 = get_xy(291)
        xtop, ytop = get_xy(13)
        xbot, ybot = get_xy(14)

        # 광대
        _, cheek_left_y = get_xy(234)
        _, cheek_right_y = get_xy(454)
        cheek_diff = abs(cheek_left_y - cheek_right_y)

        # 중심 좌표
        xnose, _ = get_xy(2)
        lip_center_x = (x1 + x2) / 2
        center_offset = abs(xnose - lip_center_x)

        # 입 계산
        mouth_width = abs(x2 - x1)
        mouth_open = abs(ybot - ytop)
        mouth_open_ratio = mouth_open / mouth_width if mouth_width else 0
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        lip_asymmetry = abs(y1 - y2)

        # 눈썹 가림
        LEFT_EYEBROW = [65, 66, 67, 68, 69]
        LEFT_EYE = 159
        eyebrow_diffs = [abs(get_y(e) - get_y(LEFT_EYE)) for e in LEFT_EYEBROW]
        avg_eyebrow_diff = sum(eyebrow_diffs) / len(eyebrow_diffs)
        eyebrow_hidden = "⭕ 보임" if avg_eyebrow_diff >= 3.5 else "❌ 가림 의심"

        # 귀 노출
        face_width = abs(get_x(454) - get_x(234))
        ear_hidden = "⭕ 보임" if face_width >= (w * 0.35) else "❌ 가림 의심"

        # 시선 추정
        def estimate_gaze(outer, inner, center_idx):
            outer_x = get_x(outer)
            inner_x = get_x(inner)
            center_x = get_x(center_idx)
            eye_width = abs(outer_x - inner_x)
            rel = abs(center_x - (outer_x + inner_x) / 2)
            return rel / eye_width if eye_width else 1.0

        gaze_left = estimate_gaze(33, 133, 468)
        gaze_right = estimate_gaze(362, 263, 473)
        gaze_result = "⭕ 정면 응시" if gaze_left < 0.25 and gaze_right < 0.25 else "❌ 시선 이탈"

        # 얼굴 정면 여부
        x_left_face = get_x(127)
        x_right_face = get_x(356)
        face_center_x = (x_left_face + x_right_face) / 2
        nose_offset = abs(xnose - face_center_x)
        face_straight = "⭕ 정면" if nose_offset < (w * 0.03) else "❌ 측면"

        # 최종 판단
        if (
            (dominant_emotion in ["neutral", "happy"]) and
            (mouth_open_ratio < 0.12) and
            (mouth_width < 60) and
            (slope <= 0.05) and
            (lip_asymmetry < 4.0) and
            (cheek_diff < 6.0) and
            (center_offset < 10.0)
        ):
            judgment = "⭕ 적합"
        else:
            judgment = "❌ 부적합"

    return {
        "result": judgment,
        "emotion": dict_emotion_kor.get(dominant_emotion, dominant_emotion),
        "입꼬리기울기": round(slope, 4) if slope else None,
        "입꼬리거리(px)": round(mouth_width, 2) if mouth_width else None,
        "입벌어짐비율": round(mouth_open_ratio, 2) if mouth_open_ratio else None,
        "입꼬리비대칭": round(lip_asymmetry, 2) if lip_asymmetry else None,
        "광대비대칭": round(cheek_diff, 2) if cheek_diff else None,
        "입중앙오프셋": round(center_offset, 2) if center_offset else None,
        "눈썹가림": eyebrow_hidden,
        "귀노출": ear_hidden,
        "시선정면": gaze_result,
        "얼굴정면": face_straight,
        "최종 판단": judgment
    }
