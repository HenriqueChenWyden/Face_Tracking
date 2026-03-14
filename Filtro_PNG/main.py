import cv2
import numpy as np


def overlay_png(background, overlay, x, y, w, h):
    """
    Sobrepõe uma imagem PNG com canal alpha (transparência)
    sobre a imagem de fundo.
    """
    if w <= 0 or h <= 0:
        return background

    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    if overlay_resized.shape[2] < 4:
        return background

    bg_h, bg_w = background.shape[:2]

    # Ajuste para não sair da tela
    if x < 0:
        overlay_resized = overlay_resized[:, -x:]
        w = overlay_resized.shape[1]
        x = 0

    if y < 0:
        overlay_resized = overlay_resized[-y:, :]
        h = overlay_resized.shape[0]
        y = 0

    if x + w > bg_w:
        overlay_resized = overlay_resized[:, :bg_w - x]
        w = overlay_resized.shape[1]

    if y + h > bg_h:
        overlay_resized = overlay_resized[:bg_h - y, :]
        h = overlay_resized.shape[0]

    if w <= 0 or h <= 0:
        return background

    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    alpha = a.astype(float) / 255.0
    alpha = np.dstack([alpha, alpha, alpha])

    roi = background[y:y + h, x:x + w].astype(float)
    overlay_rgb = overlay_rgb.astype(float)

    blended = (alpha * overlay_rgb) + ((1 - alpha) * roi)
    background[y:y + h, x:x + w] = blended.astype(np.uint8)

    return background


def iou(box_a, box_b):
    """
    Calcula o IoU (Intersection over Union) entre dois retângulos.
    box = (x, y, w, h)
    """
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def remove_duplicates(faces, threshold=0.30):
    """
    Remove detecções duplicadas mantendo a maior caixa
    quando houver muita sobreposição.
    """
    if not faces:
        return []

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    final_faces = []

    for face in faces:
        duplicated = False
        for saved in final_faces:
            if iou(face, saved) > threshold:
                duplicated = True
                break

        if not duplicated:
            final_faces.append(face)

    return final_faces


# -----------------------------
# CARREGAR CLASSIFICADORES
# -----------------------------
front_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

if front_cascade.empty():
    print("Erro ao carregar o classificador frontal.")
    exit()

if profile_cascade.empty():
    print("Erro ao carregar o classificador de perfil.")
    exit()

# -----------------------------
# CARREGAR FILTRO PNG
# -----------------------------
filter_img = cv2.imread("filtro.png", cv2.IMREAD_UNCHANGED)

if filter_img is None:
    print("Erro: não foi possível carregar 'filtro.png'.")
    print("Coloque o arquivo na mesma pasta do app1.py")
    exit()

if filter_img.shape[2] < 4:
    print("Erro: o filtro precisa ser uma imagem PNG com transparência.")
    exit()

# -----------------------------
# INICIAR WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da webcam.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detected_faces = []

    # -----------------------------
    # 1. TENTA DETECTAR ROSTO FRONTAL
    # -----------------------------
    faces_front = front_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    if len(faces_front) > 0:
        detected_faces = [tuple(face) for face in faces_front]

    else:
        # -----------------------------
        # 2. SE NÃO ACHAR FRONTAL, TENTA PERFIL
        # -----------------------------
        faces_profile = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(70, 70)
        )

        for face in faces_profile:
            detected_faces.append(tuple(face))

        # Detectar perfil do outro lado usando imagem espelhada
        gray_flipped = cv2.flip(gray, 1)
        faces_profile_flipped = profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(70, 70)
        )

        frame_width = gray.shape[1]

        for (x, y, w, h) in faces_profile_flipped:
            x_original = frame_width - x - w
            detected_faces.append((x_original, y, w, h))

        detected_faces = remove_duplicates(detected_faces, threshold=0.25)

    # -----------------------------
    # EXIBIR QUANTIDADE DE FACES
    # -----------------------------
    cv2.putText(
        frame,
        f"Faces detectadas: {len(detected_faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # -----------------------------
    # DESENHAR FILTRO
    # -----------------------------
    for (x, y, w, h) in detected_faces:
        # Retângulo do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Texto
        texto_y = y - 10 if y - 10 > 20 else y + h + 25
        cv2.putText(
            frame,
            "Rosto Detectado",
            (x, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # AJUSTE DO CHAPÉU
        # -----------------------------
        filtro_w = int(w * 1.2)
        filtro_h = int(h * 0.7)
        filtro_x = x - int((filtro_w - w) / 2)
        filtro_y = y - int(filtro_h * 0.8)

        frame = overlay_png(frame, filter_img, filtro_x, filtro_y, filtro_w, filtro_h)

    cv2.imshow("Face Tracking com Filtro PNG", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()