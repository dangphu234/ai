#!/usr/bin/env python3
"""Demo nhận dạng cảm xúc khuôn mặt trực tiếp bằng webcam (FER2013 ensemble).

Chạy:
    python webcam_demo.py                      # dùng webcam mặc định (camera 0)
    python webcam_demo.py --camera 1           # chọn camera khác
    python webcam_demo.py --video clip.mp4     # đọc từ file video thay cho webcam
    python webcam_demo.py --output demo.mp4    # vừa chạy vừa ghi video (làm bản dự phòng)

Phím tắt:  q hoặc ESC = thoát,  s = lưu ảnh chụp màn hình,  t = bật/tắt flip-TTA.
Cần cạnh file này: emotion_model_ensemble_final.keras (hoặc dùng --model).
"""
import argparse
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from collections import deque

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

print('Đang khởi động TensorFlow (lần đầu có thể mất 30-60 giây, đừng nhấn Ctrl+C)...', flush=True)
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = 48
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_VI = ['Giận', 'Ghê tởm', 'Sợ', 'Vui', 'Buồn', 'Ngạc nhiên', 'Bình thường']
# Màu BGR cho khung mặt theo cảm xúc
COLORS = [(60, 60, 230), (60, 160, 60), (200, 120, 60), (60, 200, 60),
          (200, 100, 60), (60, 200, 230), (180, 180, 180)]
FACE_MARGIN_X, FACE_MARGIN_Y = 0.10, 0.12   # giống notebook
DETECT_WIDTH = 480                           # thu nhỏ khung hình trước khi dò mặt cho nhanh
MIN_FACE_PX = 60                             # mặt nhỏ hơn (trên ảnh đã thu nhỏ) sẽ bị bỏ qua
LOW_CONFIDENCE = 0.40


# ----------------------------- nạp mô hình -----------------------------
def _strip_compile_config(obj):
    if isinstance(obj, dict):
        obj.pop('compile_config', None)
        for v in obj.values():
            _strip_compile_config(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_compile_config(v)


def load_model(path):
    """Nạp mô hình chỉ để suy luận; tự bỏ compile_config nếu lệch phiên bản Keras."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as err:
        print('Nạp trực tiếp lỗi, thử bỏ compile_config:', str(err)[:100])
    tmp = os.path.join(tempfile.gettempdir(), 'ensemble_no_compile.keras')
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == 'config.json':
                cfg = json.loads(data)
                _strip_compile_config(cfg)
                data = json.dumps(cfg).encode()
            zout.writestr(name, data)
    return tf.keras.models.load_model(tmp, compile=False, safe_mode=False)


def load_cascade():
    name = 'haarcascade_frontalface_default.xml'
    for p in (os.path.join(getattr(cv2.data, 'haarcascades', ''), name), name):
        if os.path.exists(p):
            return cv2.CascadeClassifier(p)
    print('Chưa có', name, '- đang tải từ GitHub...')
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/' + name, name)
    return cv2.CascadeClassifier(name)


# ----------------------------- xử lý ảnh -----------------------------
def preprocess_gray_face(gray_face):
    """Giống hệt hàm trong notebook: xám -> 48x48 -> [0, 1] -> (48, 48, 1)."""
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (face.astype('float32') / 255.0)[..., np.newaxis]


def find_largest_face(gray, cascade):
    """Dò mặt trên bản thu nhỏ rồi quy đổi tọa độ về khung hình gốc. Trả về (x, y, w, h) hoặc None."""
    scale = min(1.0, DETECT_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=6, minSize=(MIN_FACE_PX, MIN_FACE_PX))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return tuple(int(round(v / scale)) for v in (x, y, w, h))


def crop_face(gray, box):
    x, y, w, h = box
    mx, my = int(w * FACE_MARGIN_X), int(h * FACE_MARGIN_Y)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(gray.shape[1], x + w + mx), min(gray.shape[0], y + h + my)
    return gray[y1:y2, x1:x2]


def make_predictor(model):
    """Bọc mô hình bằng tf.function: nhanh hơn model.predict() khoảng 10 lần cho từng khung hình."""
    fn = tf.function(lambda x: model(x, training=False),
                     input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32)])
    fn(tf.zeros([1, IMG_SIZE, IMG_SIZE, 1]))      # chạy thử một lần để dựng đồ thị tính toán
    return lambda batch: fn(tf.constant(batch)).numpy()


def predict_face(predictor, face_gray, use_tta):
    batch = [preprocess_gray_face(face_gray)]
    if use_tta:
        batch.append(preprocess_gray_face(cv2.flip(face_gray, 1)))
    return predictor(np.stack(batch)).mean(axis=0)


# ----------------------------- vẽ giao diện -----------------------------
_FONT_CACHE = {}


def get_font(size):
    """Tìm font hỗ trợ tiếng Việt trên Windows / macOS / Linux; không có thì trả về None."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for name in ('arial.ttf', 'segoeui.ttf', 'Arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
                 'DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    _FONT_CACHE[size] = font
    return font


PANEL_W = 290


def draw_ui(frame, box, probs, fps, use_tta, status):
    """Vẽ khung mặt + nhãn lên video và ghép thêm cột xác suất bên phải (không che mặt)."""
    h, w = frame.shape[:2]
    video = frame.copy()
    panel = np.full((h, PANEL_W, 3), 34, np.uint8)
    t_video, t_panel = [], []          # (x, y, chuỗi, cỡ chữ, màu RGB)
    top = int(np.argmax(probs)) if probs is not None else None

    # --- Khung mặt và nhãn trên video ---
    if box is not None and probs is not None:
        x, y, bw, bh = box
        color = COLORS[top]
        cv2.rectangle(video, (x, y), (x + bw, y + bh), color, 3)
        tag = '{} ({}) {:.0f}%'.format(LABELS[top], LABELS_VI[top], probs[top] * 100)
        if probs[top] < LOW_CONFIDENCE:
            tag += ' ?'
        ty = y - 32 if y > 38 else y + bh + 6
        cv2.rectangle(video, (x, ty), (x + max(bw, 250), ty + 28), color, -1)
        t_video.append((x + 8, ty + 3, tag, 18, (255, 255, 255)))
    if status:
        cv2.rectangle(video, (0, h - 40), (w, h), (30, 30, 30), -1)
        t_video.append((12, h - 33, status, 18, (80, 220, 255)))

    # --- Cột xác suất ---
    t_panel.append((16, 14, 'Xác suất cảm xúc', 20, (255, 255, 255)))
    for i, name in enumerate(LABELS_VI):
        y = 62 + i * 46
        p = float(probs[i]) if probs is not None else 0.0
        hi = (i == top)
        txt = (255, 255, 255) if hi else (175, 175, 175)
        t_panel.append((16, y, name, 19, txt))
        t_panel.append((PANEL_W - 66, y, '{:.0f}%'.format(p * 100), 19, txt))
        cv2.rectangle(panel, (16, y + 28), (PANEL_W - 16, y + 36), (60, 60, 60), -1)
        cv2.rectangle(panel, (16, y + 28), (16 + int(p * (PANEL_W - 32)), y + 36), COLORS[i] if hi else (140, 140, 140), -1)
    t_panel.append((16, h - 78, 'FPS {:.0f}   |   TTA {}'.format(fps, 'bật' if use_tta else 'tắt'), 16, (200, 200, 200)))
    t_panel.append((16, h - 52, 'q: thoát   s: chụp   t: TTA', 16, (150, 150, 150)))

    out = np.hstack([video, panel])

    # --- Vẽ chữ bằng PIL (có tiếng Việt); không có font thì dùng cv2 bỏ dấu ---
    items = t_video + [(x + w, y, s_, sz, c) for x, y, s_, sz, c in t_panel]
    if get_font(16) is not None:
        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for x, y, s_, sz, rgb in items:
            d.text((x, y), s_, font=get_font(sz), fill=rgb)
        out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        for x, y, s_, sz, rgb in items:
            cv2.putText(out, s_.encode('ascii', 'ignore').decode(), (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, rgb[::-1], 1, cv2.LINE_AA)
    return out


# ----------------------------- vòng lặp chính -----------------------------
def main():
    ap = argparse.ArgumentParser(description='Demo cảm xúc khuôn mặt bằng webcam')
    ap.add_argument('--model', default='emotion_model_ensemble_final.keras')
    ap.add_argument('--camera', type=int, default=0, help='chỉ số camera (mặc định 0)')
    ap.add_argument('--video', default=None, help='đọc từ file video thay cho webcam')
    ap.add_argument('--output', default=None, help='ghi video kết quả ra file .mp4')
    ap.add_argument('--smooth', type=int, default=8, help='số khung hình dùng để làm mượt (mặc định 8)')
    ap.add_argument('--every', type=int, default=1, help='dự đoán mỗi N khung hình (tăng nếu máy chậm)')
    ap.add_argument('--tta', action='store_true', help='bật flip-TTA (chính xác hơn nhưng chậm gấp đôi)')
    ap.add_argument('--no-mirror', action='store_true', help='không lật gương khung hình webcam')
    ap.add_argument('--no-window', action='store_true', help='không mở cửa sổ (chạy ngầm, dùng với --output)')
    ap.add_argument('--height', type=int, default=720, help='chiều cao khung hình hiển thị (mặc định 720)')
    ap.add_argument('--max-frames', type=int, default=0, help='dừng sau N khung hình (0 = không giới hạn)')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit('Không thấy mô hình: ' + args.model)
    print('Đang nạp mô hình...')
    model = load_model(args.model)
    predictor = make_predictor(model)
    cascade = load_cascade()
    if cascade.empty():
        raise SystemExit('Không nạp được haarcascade_frontalface_default.xml')

    if args.video:
        cap = cv2.VideoCapture(args.video)
        mirror = False
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(args.camera)
        mirror = not args.no_mirror
    if not cap.isOpened():
        raise SystemExit('Không mở được ' + ('video ' + args.video if args.video
                                              else 'camera {} (kiểm tra quyền camera / đóng ứng dụng khác đang dùng)'.format(args.camera)))

    use_tta = args.tta
    history = deque(maxlen=max(1, args.smooth))
    writer = None
    last_box, last_probs = None, None
    lost = 0
    frame_i, fps, t_prev = 0, 0.0, time.time()
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if src_fps and src_fps > 1 else 20.0
    win = 'Nhan dang cam xuc (q = thoat, s = chup, t = TTA)'
    window_ready = False
    print('Bắt đầu. Nhấn q hoặc ESC để thoát.')

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        frame_i += 1

        if (frame_i - 1) % max(1, args.every) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = find_largest_face(gray, cascade)
            if box is None:
                lost += 1
                if lost > 5:                       # mất mặt vài khung hình thì xóa lịch sử làm mượt
                    history.clear()
                    last_box, last_probs = None, None
            else:
                lost = 0
                face = crop_face(gray, box)
                history.append(predict_face(predictor, face, use_tta))
                last_box, last_probs = box, np.mean(history, axis=0)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        status = '' if last_box is not None else 'Không thấy khuôn mặt - hãy nhìn thẳng vào camera'
        scale = args.height / frame.shape[0]
        disp = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if abs(scale - 1) > 0.01 else frame
        box_d = tuple(int(round(v * scale)) for v in last_box) if last_box is not None else None
        shown = draw_ui(disp, box_d, last_probs, fps, use_tta, status)

        if args.output:
            if writer is None:
                h, w = shown.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w, h))
            writer.write(shown)

        if not args.no_window:
            try:
                if not window_ready:      # cửa sổ co giãn được, giữ đúng tỉ lệ khi phóng to/thu nhỏ
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(win, shown.shape[1], shown.shape[0])
                    window_ready = True
                cv2.imshow(win, shown)
            except cv2.error:
                raise SystemExit('OpenCV không mở được cửa sổ (đang dùng bản headless?). '
                                 'Cài lại: pip uninstall opencv-python-headless && pip install opencv-python, '
                                 'hoặc chạy với --no-window --output demo.mp4')
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                name = 'chup_{}.png'.format(time.strftime('%H%M%S'))
                cv2.imwrite(name, shown)
                print('Đã lưu', name)
            if key == ord('t'):
                use_tta = not use_tta
                history.clear()
        if args.max_frames and frame_i >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print('Đã ghi video:', args.output)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()