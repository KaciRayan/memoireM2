import os
import cv2
import numpy as np
import math

def analyse_composition(original_filename):
    path = os.path.join("app", "static/images", original_filename)
    image = cv2.imread(path)

    analyses = []
    analyses.append(detect_regle_tiers(image))
    analyses.append(detect_composition_diagonale(image))
    analyses.append(detect_centrage(image))
    # analyses.append(detect_composition_L(image))
    # analyses.append(detect_nombre_or(image))

    analyses.append(analyse_flou(image, original_filename))

    # + d'autres règles à venir

    règles_valides = [a for a in analyses if a["evaluation"]]

    if règles_valides:
        return règles_valides
    else:
        return []

def detect_regle_tiers(image):
    height, width = image.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    threshMap = cv2.threshold(saliencyMap, 200, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tier_points = [
        (int(width * 1/3), int(height * 1/3)),
        (int(width * 2/3), int(height * 1/3)),
        (int(width * 1/3), int(height * 2/3)),
        (int(width * 2/3), int(height * 2/3))
    ]

    rayon_tol = min(width, height) * 0.1  # tolérance

    match_found = False
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        for (tx, ty) in tier_points:
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if dist < rayon_tol:
                match_found = True
                break

    evaluation = match_found
    description = (
        "Un ou plusieurs éléments visuellement saillants sont situés à proximité "
        "des points d’intersection des lignes de tiers."
        if match_found else
        "Les éléments les plus saillants ne semblent pas alignés avec les points clés de la règle des tiers."
    )
    interpretation = (
        "Cela suggère une composition harmonieuse qui guide naturellement le regard."
        if match_found else
        "L'artiste semble s’éloigner des règles classiques de composition, ce qui peut traduire une volonté de déséquilibre ou d’originalité."
    )

    visuel = image.copy()
    for (tx, ty) in tier_points:
        cv2.circle(visuel, (tx, ty), 10, (0, 255, 0), -1)  # cercles verts sur les points de tiers

    tiers_x1 = width // 3
    tiers_x2 = 2 * width // 3
    tiers_y1 = height // 3
    tiers_y2 = 2 * height // 3

    line_color = (255, 0, 0)  # bleu en BGR
    thickness = 1

    cv2.line(visuel, (tiers_x1, 0), (tiers_x1, height), line_color, thickness)
    cv2.line(visuel, (tiers_x2, 0), (tiers_x2, height), line_color, thickness)

    cv2.line(visuel, (0, tiers_y1), (width, tiers_y1), line_color, thickness)
    cv2.line(visuel, (0, tiers_y2), (width, tiers_y2), line_color, thickness)

    visuel_filename = "tiers_visuel.jpg"
    visuel_path = os.path.join("app", "static/images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    return {
        "regle": "Règle des tiers",
        "evaluation": evaluation,
        "description": description,
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

def detect_composition_diagonale(image):
    height, width = image.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap, 200, 255, cv2.THRESH_BINARY)[1]

    ys, xs = np.where(threshMap > 0)
    coords = np.column_stack((xs, ys))

    if len(coords) < 2:
        return {
            "regle": "Composition diagonale",
            "evaluation": False,
            "description": "Trop peu de zones saillantes pour analyser la composition.",
            "interpretation": "L’image semble peu contrastée ou dépourvue de points d’attention forts.",
            "visuel_path": None
        }

    coords_mean = np.mean(coords, axis=0)
    coords_centered = coords - coords_mean
    _, _, vh = np.linalg.svd(coords_centered)
    principal_axis = vh[0]  

    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg % 180  

    tolerance = 20
    is_diagonal = (
        abs(angle_deg - 45) < tolerance or abs(angle_deg - 135) < tolerance
    )

    visuel = image.copy()
    pt1 = tuple(np.int32(coords_mean - 200 * principal_axis))
    pt2 = tuple(np.int32(coords_mean + 200 * principal_axis))
    cv2.line(visuel, pt1, pt2, (0, 0, 255), 2) 

    for (x, y) in coords:
        cv2.circle(visuel, (x, y), 2, (0, 255, 0), -1) 

    visuel_filename = "diagonale_visuel.jpg"
    visuel_path = os.path.join("app", "static/images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    description = (
        f"L’axe principal des zones saillantes forme un angle de {angle_deg:.1f}° avec l’horizontale."
    )
    interpretation = (
        "Cela suggère une composition diagonale, qui insuffle un dynamisme et une tension visuelle."
        if is_diagonal else
        "L’image ne présente pas de direction diagonale dominante dans la répartition des masses saillantes."
    )

    return {
        "regle": "Composition diagonale",
        "evaluation": is_diagonal,
        "description": description,
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

def detect_centrage(image):
    height, width = image.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    thresh = cv2.threshold(saliencyMap, 200, 255, cv2.THRESH_BINARY)[1]

    ys, xs = np.where(thresh > 0)
    if len(xs) < 2:
        return {
            "regle": "Centrage",
            "evaluation": False,
            "description": "Pas assez de zones saillantes pour analyser le centrage.",
            "interpretation": "L’image semble peu structurée autour d’un axe central.",
            "visuel_path": None
        }

    cx = np.mean(xs)
    center_x = width / 2
    tol = width * 0.1

    evaluation = abs(cx - center_x) < tol

    visuel = image.copy()
    cv2.line(visuel, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 1) 
    cv2.circle(visuel, (int(cx), int(np.mean(ys))), 8, (0, 255, 0), -1) 

    visuel_filename = "centrage_visuel.jpg"
    visuel_path = os.path.join("app", "static", "images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    description = "Le centre de gravité des zones saillantes est proche de l’axe central vertical."
    interpretation = (
        "Cette composition centrée crée une impression de stabilité, de frontalité ou de symétrie solennelle."
        if evaluation else
        "L’artiste semble avoir décentré les masses, peut-être pour suggérer un déséquilibre ou une dynamique latérale."
    )

    return {
        "regle": "Centrage",
        "evaluation": evaluation,
        "description": description,
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

# def detect_composition_L(image):
    height, width = image.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    thresh = cv2.threshold(saliencyMap, 200, 255, cv2.THRESH_BINARY)[1]

    top_half = thresh[:height//2, :]
    bottom_half = thresh[height//2:, :]
    left_half = thresh[:, :width//2]
    right_half = thresh[:, width//2:]

    sal_top = np.sum(top_half)
    sal_bottom = np.sum(bottom_half)
    sal_left = np.sum(left_half)
    sal_right = np.sum(right_half)

    total = sal_top + sal_bottom + sal_left + sal_right
    if total == 0:
        return {
            "regle": "Composition en L",
            "evaluation": False,
            "description": "Aucune masse saillante claire détectée.",
            "interpretation": "Pas de structuration visuelle forte détectée dans un agencement en L.",
            "visuel_path": None
        }

    p_left_bottom = (sal_left + sal_bottom) / total
    p_right_bottom = (sal_right + sal_bottom) / total

    evaluation = p_left_bottom > 0.6 or p_right_bottom > 0.6
    direction = "gauche + bas" if p_left_bottom > p_right_bottom else "droite + bas"

    visuel = image.copy()
    cv2.rectangle(visuel, (0, height//2), (width//2, height), (0, 255, 0), 2)
    cv2.rectangle(visuel, (width//2, height//2), (width, height), (255, 255, 0), 2)

    visuel_filename = "composition_L_visuel.jpg"
    visuel_path = os.path.join("app", "static", "images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    description = f"La majorité des zones saillantes se situent dans la partie {direction} de l’image."
    interpretation = (
        "La structure en L guide naturellement le regard de haut en bas et d’un côté à l’autre, "
        "créant une base solide et une lecture confortable."
        if evaluation else
        "La répartition des masses ne correspond pas à une structure en L classique."
    )

    return {
        "regle": "Composition en L",
        "evaluation": evaluation,
        "description": description,
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

# def detect_nombre_or(image):
    height, width = image.shape[:2]
    phi = 0.618

    golden_x = [int(width * phi), int(width * (1 - phi))]
    golden_y = [int(height * phi), int(height * (1 - phi))]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    thresh = cv2.threshold(saliencyMap, 200, 255, cv2.THRESH_BINARY)[1]

    ys, xs = np.where(thresh > 0)
    if len(xs) < 2:
        return {
            "regle": "Nombre d’or",
            "evaluation": False,
            "description": "Pas assez de points saillants pour évaluer la composition.",
            "interpretation": "L’image semble peu structurée selon des proportions classiques.",
            "visuel_path": None
        }

    cx, cy = np.mean(xs), np.mean(ys)

    match_found = any(
        abs(cx - gx) < width * 0.05 and abs(cy - gy) < height * 0.05
        for gx in golden_x
        for gy in golden_y
    )

    visuel = image.copy()
    for gx in golden_x:
        for gy in golden_y:
            cv2.circle(visuel, (gx, gy), 8, (0, 255, 255), -1)  # jaune = points d’or
    cv2.circle(visuel, (int(cx), int(cy)), 8, (0, 0, 255), -1)  # rouge = centre saillant

    visuel_filename = "nombre_or_visuel.jpg"
    visuel_path = os.path.join("app", "static", "images", visuel_filename)
    cv2.imwrite(visuel_path, visuel)

    description = (
        "Le centre de gravité des zones saillantes est proche d’un des points d’or théoriques."
    )
    interpretation = (
        "Cette structure évoque un équilibre visuel très recherché dans l’art et la photographie, "
        "fondé sur des proportions naturelles harmonieuses."
        if match_found else
        "La composition semble s’éloigner des canons du nombre d’or."
    )

    return {
        "regle": "Nombre d’or (approximatif)",
        "evaluation": match_found,
        "description": description,
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }

def analyse_flou(image, original_filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    laplacian_clipped = np.clip(laplacian_abs, 0, 50)  # plafonne les extrêmes
    laplacian_norm = (laplacian_clipped / 50 * 255).astype(np.uint8)

    variance = laplacian.var()

    heatmap = cv2.applyColorMap(laplacian_norm, cv2.COLORMAP_JET)

    legend_height = heatmap.shape[0]
    legend_width = 40
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

    for i in range(legend_height):
        value = int((legend_height - i - 1) / legend_height * 255)
        color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        legend[i, :] = color

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, "net", (3, 25), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "flou", (3, legend_height - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    final_visuel = np.hstack((heatmap, legend))

    visuel_filename = f"flou_visuel_{original_filename}"
    visuel_path = os.path.join("app", "static", "images", visuel_filename)
    cv2.imwrite(visuel_path, final_visuel)

    if variance > 300:
        interpretation = (
            "L’image présente un haut niveau de netteté. "
            "Les contours sont bien définis."
        )
    elif 100 < variance <= 300:
        interpretation = (
            "L’image montre un équilibre entre zones nettes et floues. "
            "Cela crée une hiérarchie visuelle intéressante."
        )
    else:
        interpretation = (
            "L’image semble globalement floue. Cela peut traduire un effet artistique voulu, "
            "ou nécessiter plus de définition dans les zones importantes."
        )

    return {
        "regle": "Analyse de netteté",
        "evaluation": True,
        "description": "La répartition de la netteté a été analysée par la variance du Laplacien.",
        "interpretation": interpretation,
        "visuel_path": visuel_filename
    }