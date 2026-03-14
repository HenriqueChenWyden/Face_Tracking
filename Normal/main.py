import cv2

# ==============================
# CARREGAR CLASSIFICADORES
# ==============================
face_frontal = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

if face_frontal.empty():
    print("Erro ao carregar haarcascade_frontalface_default.xml")
    exit()

if face_profile.empty():
    print("Erro ao carregar haarcascade_profileface.xml")
    exit()

# ==============================
# ABRIR WEBCAM
# ==============================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

# ==============================
# FUNÇÕES PARA REMOVER DUPLICATAS
# ==============================
def calcular_iou(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    largura_intersecao = max(0, xb - xa)
    altura_intersecao = max(0, yb - ya)
    area_intersecao = largura_intersecao * altura_intersecao

    area_r1 = w1 * h1
    area_r2 = w2 * h2
    area_uniao = area_r1 + area_r2 - area_intersecao

    if area_uniao == 0:
        return 0

    return area_intersecao / area_uniao

def remover_duplicados(lista_retangulos, limite_iou=0.35):
    resultado = []

    for atual in lista_retangulos:
        duplicado = False

        for salvo in resultado:
            if calcular_iou(atual, salvo) > limite_iou:
                duplicado = True
                break

        if not duplicado:
            resultado.append(atual)

    return resultado

# ==============================
# LOOP PRINCIPAL
# ==============================
while True:
    ok, frame = camera.read()

    if not ok:
        print("Erro ao capturar frame da webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ==============================
    # DETECÇÃO FRONTAL
    # ==============================
    faces_frontal = face_frontal.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=7,
        minSize=(100, 100)
    )

    # ==============================
    # DETECÇÃO PERFIL NORMAL
    # ==============================
    faces_profile = face_profile.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # ==============================
    # DETECÇÃO PERFIL INVERTIDO
    # ==============================
    gray_invertido = cv2.flip(gray, 1)
    faces_profile_invertido = face_profile.detectMultiScale(
        gray_invertido,
        scaleFactor=1.3,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # ==============================
    # JUNTAR TODAS AS DETECÇÕES
    # ==============================
    deteccoes = []

    for (x, y, w, h) in faces_frontal:
        deteccoes.append((x, y, w, h))

    for (x, y, w, h) in faces_profile:
        deteccoes.append((x, y, w, h))

    largura_frame = frame.shape[1]
    for (x, y, w, h) in faces_profile_invertido:
        x_original = largura_frame - x - w
        deteccoes.append((x_original, y, w, h))

    # Remove retângulos repetidos
    deteccoes_finais = remover_duplicados(deteccoes, limite_iou=0.35)

    # ==============================
    # DESENHAR RESULTADOS
    # ==============================
    total_faces = len(deteccoes_finais)

    if total_faces > 0:
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
        f"Quantidade de faces: {total_faces}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    for (x, y, w, h) in deteccoes_finais:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.putText(
            frame,
            "Face Tracking",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.imshow("Deteccao de Rosto Melhorada", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()