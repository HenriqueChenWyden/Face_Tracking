import cv2
import numpy as np

# =========================
# CARREGAR CLASSIFICADORES
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# =========================
# CARREGAR MÁSCARA
# =========================
mask_path = "mascara.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

if mask is None:
    print("Erro: não foi possível carregar 'mascara.png'")
    exit()

if mask.shape[2] != 4:
    print("Erro: a imagem precisa ser PNG com transparência.")
    exit()

# =========================
# FUNÇÃO PARA SOBREPOR PNG
# =========================
def overlay_png(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background

    overlay_rgb = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3] / 255.0

    roi = background[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] + (1 - alpha) * roi[:, :, c])

    background[y1:y2, x1:x2] = roi
    return background

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    detected_faces = []

    if len(faces) == 0:
        profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in profiles:
            detected_faces.append((x, y, w, h))

        gray_flipped = cv2.flip(gray, 1)
        profiles_flipped = profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in profiles_flipped:
            x_original = gray.shape[1] - x - w
            detected_faces.append((x_original, y, w, h))
    else:
        detected_faces = faces

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Rosto Detectado",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # =========================
        # AJUSTE FINO DA MÁSCARA (substituir)
        # =========================
        face_width = w
        scale_w = 1.4  # aumente/reduza este valor para ajustar o tamanho
        mask_w = int(face_width * scale_w)

        # preservar proporção original da máscara (altura/largura)
        mask_aspect = mask.shape[0] / mask.shape[1]
        mask_h = int(mask_w * mask_aspect)

        # centralizar horizontalmente e deslocar verticalmente (ajuste conforme necessário)
        mask_x = x - (mask_w - w) // 2
        mask_y = y - int(mask_h * 0.20)  # move a máscara para cima; ajuste o 0.35 se necessário

        frame = overlay_png(frame, mask, mask_x, mask_y, mask_w, mask_h)

        cv2.imshow("Filtro Expert - Po Kung Fu Panda", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()