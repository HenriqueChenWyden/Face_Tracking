import cv2
import math

# =====================================
# CARREGAR CLASSIFICADORES
# =====================================
face_frontal = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_frontal.empty():
    print("Erro ao carregar haarcascade_frontalface_default.xml")
    exit()

if face_profile.empty():
    print("Erro ao carregar haarcascade_profileface.xml")
    exit()

if eye_cascade.empty():
    print("Erro ao carregar haarcascade_eye.xml")
    exit()

# =====================================
# FUNÇÕES AUXILIARES
# =====================================
def calcular_iou(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union


def centro_retangulo(r):
    x, y, w, h = r
    return (x + w / 2, y + h / 2)


def distancia_centros(r1, r2):
    c1x, c1y = centro_retangulo(r1)
    c2x, c2y = centro_retangulo(r2)
    return math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)


def retangulo_dentro(r1, r2, margem=10):
    # retorna True se r1 estiver praticamente dentro de r2
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    return (
        x1 >= x2 - margem and
        y1 >= y2 - margem and
        x1 + w1 <= x2 + w2 + margem and
        y1 + h1 <= y2 + h2 + margem
    )


def remover_duplicados_avancado(lista_retangulos):
    if not lista_retangulos:
        return []

    # ordenar do maior para o menor
    lista = sorted(lista_retangulos, key=lambda r: r[2] * r[3], reverse=True)
    resultado = []

    for atual in lista:
        manter = True

        for salvo in resultado:
            iou = calcular_iou(atual, salvo)
            dist = distancia_centros(atual, salvo)

            x, y, w, h = atual
            sx, sy, sw, sh = salvo

            area_atual = w * h
            area_salvo = sw * sh

            # 1. remove se estiver dentro de outro maior
            if retangulo_dentro(atual, salvo, margem=15):
                manter = False
                break

            # 2. remove se a sobreposição for alta
            if iou > 0.25:
                manter = False
                break

            # 3. remove se o centro for muito próximo e o atual for menor
            limite_dist = min(sw, sh) * 0.35
            if dist < limite_dist and area_atual < area_salvo * 0.75:
                manter = False
                break

        if manter:
            resultado.append(atual)

    return resultado


def limitar_olhos(olhos, max_olhos=2):
    olhos_ordenados = sorted(olhos, key=lambda o: o[2] * o[3], reverse=True)
    return olhos_ordenados[:max_olhos]

# =====================================
# ABRIR WEBCAM
# =====================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

# =====================================
# LOOP PRINCIPAL
# =====================================
while True:
    ok, frame = camera.read()

    if not ok:
        print("Erro ao capturar imagem da webcam.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # =====================================
    # DETECÇÃO FRONTAL
    # =====================================
    faces_frontal = face_frontal.detectMultiScale(
        gray_blur,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(120, 120)
    )

    # =====================================
    # DETECÇÃO PERFIL
    # =====================================
    faces_profile = face_profile.detectMultiScale(
        gray_blur,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(120, 120)
    )

    # =====================================
    # PERFIL INVERTIDO
    # =====================================
    gray_flip = cv2.flip(gray_blur, 1)
    faces_profile_flip = face_profile.detectMultiScale(
        gray_flip,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(120, 120)
    )

    deteccoes = []

    for (x, y, w, h) in faces_frontal:
        deteccoes.append((x, y, w, h))

    for (x, y, w, h) in faces_profile:
        deteccoes.append((x, y, w, h))

    largura_frame = frame.shape[1]
    for (x, y, w, h) in faces_profile_flip:
        x_original = largura_frame - x - w
        deteccoes.append((x_original, y, w, h))

    # remove duplicações e subdetecções
    deteccoes_finais = remover_duplicados_avancado(deteccoes)

    # =====================================
    # TEXTO
    # =====================================
    if len(deteccoes_finais) > 0:
        cv2.putText(
            frame,
            "Rosto Detectado",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "Nenhum rosto detectado",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {len(deteccoes_finais)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # =====================================
    # DESENHAR CÍRCULO DO ROSTO E DOS OLHOS
    # =====================================
    for (x, y, w, h) in deteccoes_finais:
        centro_x = x + w // 2
        centro_y = y + h // 2
        raio_rosto = int(min(w, h) * 0.50)

        cv2.circle(frame, (centro_x, centro_y), raio_rosto, (255, 0, 0), 3)

        cv2.putText(
            frame,
            "Face Tracking",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # procurar olhos só na parte de cima do rosto
        roi_gray = gray_blur[y:y + int(h * 0.55), x:x + w]
        roi_color = frame[y:y + int(h * 0.55), x:x + w]

        olhos = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=12,
            minSize=(25, 25),
            maxSize=(80, 80)
        )

        olhos = limitar_olhos(olhos, max_olhos=2)

        for (ex, ey, ew, eh) in olhos:
            olho_cx = ex + ew // 2
            olho_cy = ey + eh // 2
            raio_olho = int(min(ew, eh) * 0.45)

            cv2.circle(
                roi_color,
                (olho_cx, olho_cy),
                raio_olho,
                (0, 255, 255),
                2
            )

    cv2.imshow("Filtro de Rosto Corrigido", frame)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break

camera.release()
cv2.destroyAllWindows()