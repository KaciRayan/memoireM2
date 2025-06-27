import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def analyse_couleurs(original_filename):
    path = os.path.join("app", "static/images", original_filename)
    image = cv2.imread(path)

    analyses = []
    analyses.append(analyse_saturation(image, original_filename))
    analyses.append(analyse_temperature_masses(image, original_filename))
    analyses.append(analyse_palette_harmonique(image, original_filename))
    # + d'autres règles à venir

    return analyses

def analyse_temperature_masses(image, original_filename):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    chaud_mask = cv2.inRange(h, 0, 30)
    froid_mask = cv2.inRange(h, 90, 130)

    chaud_contours, _ = cv2.findContours(chaud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    froid_contours, _ = cv2.findContours(froid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visuel = image.copy()
    overlay = np.zeros_like(image)
    cv2.drawContours(overlay, chaud_contours, -1, (100, 100, 255), -1)
    cv2.drawContours(overlay, froid_contours, -1, (255, 200, 200), -1)

    alpha = 0.5  # transparence douce
    visuel = cv2.addWeighted(overlay, alpha, visuel, 1 - alpha, 0)

    visuel_filename = f"temperature_masses_{original_filename}"
    visuel_path = os.path.join("app", "static", "images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    # Compter les pixels
    chaud_pixels = np.count_nonzero(chaud_mask)
    froid_pixels = np.count_nonzero(froid_mask)
    total = chaud_pixels + froid_pixels

    if total == 0:
        interpretation = (
            "Aucune zone chaude ou froide significative n’a été détectée. "
            "L’image semble dominée par des teintes neutres ou peu saturées."
        )
    else:
        chaud_ratio = chaud_pixels / total
        froid_ratio = froid_pixels / total

        if chaud_ratio > 0.7:
            interpretation = "L’image est dominée par des masses chaudes, ce qui crée une atmosphère dynamique ou intense."
        elif froid_ratio > 0.7:
            interpretation = "L’image est dominée par des masses froides, ce qui crée une sensation de calme ou de retrait."
        else:
            interpretation = "L’image présente un équilibre visuel intéressant entre masses chaudes et froides."

    return {
        "titre": "Répartition des couleurs chaudes et froides",
        "description": "Les zones rouges représentent les masses chaudes, les zones bleues les masses froides.",
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

def analyse_saturation(image, original_filename):
    # Convertir en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    saturation_mean = np.mean(s) / 255.0

    seuil_faible = 0.3
    seuil_forte = 0.7

    # pixels très saturés
    saturation_ratio_forte = np.count_nonzero(s > seuil_forte * 255) / s.size

    if saturation_mean < seuil_faible:
        commentaire_saturation = "L’image est globalement peu saturée, avec une palette plutôt terne."
    elif saturation_mean > seuil_forte:
        commentaire_saturation = "L’image présente une saturation globale élevée, attention à l’effet visuel trop criard."
    else:
        commentaire_saturation = "L’image a une saturation moyenne bien dosée."

    if saturation_ratio_forte > 0.6:
        commentaire_repartition = "Les couleurs très saturées occupent une grande partie de l’image, ce qui peut fatiguer l’œil."
    elif saturation_ratio_forte < 0.2:
        commentaire_repartition = "Les couleurs très saturées sont peu présentes, ce qui donne une image douce mais parfois sans accent visuel."
    else:
        commentaire_repartition = "La répartition des couleurs saturées est équilibrée."

    commentaire = f"{commentaire_saturation} {commentaire_repartition}"

    small = cv2.resize(image, (100, 100))
    data = small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=8, n_init='auto')
    kmeans.fit(data)
    centers = kmeans.cluster_centers_.astype(int)

    centers_hsv = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_BGR2HSV)[0]
    couleurs_vives = []
    couleurs_ternes = []
    for i, hsv_color in enumerate(centers_hsv):
        if hsv_color[1] > seuil_faible * 255:
            couleurs_vives.append(centers[i])
        else:
            couleurs_ternes.append(centers[i])

    swatch_height = 50
    swatch_width = 100
    width_img = swatch_width * 8
    height_img = swatch_height * 2 + 90
    palette = np.ones((height_img, width_img, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    thickness = 1
    margin_top = 30
    line_spacing = 5
    section_spacing = 35  # Espace entre les deux groupes de couleurs

    for i, color in enumerate(couleurs_vives):
        x1, y1 = i * swatch_width, margin_top
        x2, y2 = x1 + swatch_width, y1 + swatch_height
        palette[y1:y2, x1:x2] = color
        cv2.rectangle(palette, (x1, y1), (x2 - 1, y2 - 1), (100, 100, 100), 1)


    y_offset = margin_top + swatch_height + section_spacing
    for i, color in enumerate(couleurs_ternes):
        x1, y1 = i * swatch_width, y_offset
        x2, y2 = x1 + swatch_width, y1 + swatch_height
        palette[y1:y2, x1:x2] = color
        cv2.rectangle(palette, (x1, y1), (x2 - 1, y2 - 1), (100, 100, 100), 1)

    cv2.putText(palette, "Couleurs vives :", (5, margin_top - line_spacing), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(palette, "Couleurs ternes :", (5, y_offset - line_spacing), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Sauvegarde
    visuel_filename = f"saturation_palette_{original_filename}"
    visuel_path = os.path.join("app", "static/images", visuel_filename)
    cv2.imwrite(visuel_path, palette)

    return {
        "titre": "Analyse de la saturation",
        "description": "Palette de couleurs extraites de l’image. Répartition basée sur la saturation.",
        "interpretation": commentaire,
        "visuel_path": visuel_filename
    }

def analyse_palette_harmonique(image, original_filename):

    small = cv2.resize(image, (100, 100))
    data = small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=8, n_init='auto')
    kmeans.fit(data)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    centers_hsv = cv2.cvtColor(np.array([centers]), cv2.COLOR_BGR2HSV)[0]
    couleurs_filtrees = []
    hues = []
    for i, hsv in enumerate(centers_hsv):
        h, s, v = hsv
        if 40 < v < 240:  
            couleurs_filtrees.append(centers[i])
            hues.append(h * 2)

    if not couleurs_filtrees:
        couleurs_filtrees = centers
        hues = [h * 2 for h in centers_hsv[:, 0]]

    size = 600
    radius = 250
    thickness = 50
    center = (size // 2, size // 2)
    wheel = np.ones((size, size, 3), dtype=np.uint8) * 255 

    for y in range(size):
        for x in range(size):
            dx = x - center[0]
            dy = y - center[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if radius - thickness <= distance <= radius:
                angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360
                hue = angle / 2
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                wheel[y, x] = bgr_color

    for i, color in enumerate(couleurs_filtrees):
        hue = hues[i]
        angle_rad = np.deg2rad(hue)
        x_end = int(center[0] + (radius - thickness // 2) * np.cos(angle_rad))
        y_end = int(center[1] - (radius - thickness // 2) * np.sin(angle_rad))

        cv2.line(wheel, center, (x_end, y_end), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(wheel, (x_end, y_end), 12, (255, 255, 255), -1)
        cv2.circle(wheel, (x_end, y_end), 10, color.tolist(), -1)

    hues_sorted = np.sort(hues).astype(int)
    n = len(hues_sorted)

    # Calcul des écarts angulaires 
    ecarts = [abs(hues_sorted[(i+1)%n] - hues_sorted[i]) for i in range(n)]
    ecarts = [e if e <= 180 else 360 - e for e in ecarts]  # corriger les cas circulaires

    type_palette = "indéterminée"
    interpretation = "Aucune harmonie chromatique claire n’a pu être identifiée à partir des teintes extraites."

    if max(ecarts) < 60:
        type_palette = "analogue"
        interpretation = "La palette utilise des couleurs analogues, proches sur la roue chromatique, créant une ambiance douce et harmonieuse."
    elif any(170 <= e <= 190 for e in ecarts):
        type_palette = "complementaire"
        interpretation = "La palette utilise des couleurs complémentaires, opposées sur la roue chromatique, générant une tension visuelle forte et dynamique."
    elif all(any(abs(e - angle) < 30 for angle in [120, 240]) for e in ecarts) and n >= 3:
        type_palette = "triadique"
        interpretation = "La palette présente une structure triadique, avec trois couleurs espacées d’environ 120°, apportant équilibre et vivacité."
    else:
        type_palette = "dispersee"
        interpretation = "La palette est dispersée, sans schéma harmonique clair, ce qui peut produire un effet éclectique ou désorganisé."

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text = f"Harmonie : {type_palette.capitalize()}"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = (wheel.shape[1] - text_size[0]) // 2
    text_y = 40 

    cv2.putText(wheel, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    visuel_filename = f"harmonie_chromatique_{original_filename}"
    visuel_path = os.path.join("app", "static/images", visuel_filename)
    cv2.imwrite(visuel_path, wheel)

    return {
        "titre": "Analyse d'harmonie chromatique",
        "description": "Les traits indiquent les couleurs dominantes de l’œuvre, projetées sur un cercle chromatique artistique.",
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }