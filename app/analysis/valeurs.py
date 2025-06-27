import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.patches import Patch

def analyse_valeurs(gray_image, original_filename):

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    total_pixels = np.sum(hist)

    # Répartition des pixels par plage de valeur
    very_dark = np.sum(hist[0:52])
    dark = np.sum(hist[52:103])
    medium = np.sum(hist[103:154])
    light = np.sum(hist[154:205])
    very_light = np.sum(hist[205:256])

    categories = {
        "très sombres": very_dark,
        "sombres": dark,
        "moyens": medium,
        "clairs": light,
        "très clairs": very_light
    }

    dominant = max(categories, key=categories.get)

    # Pourcentage 
    percentages = {
        "très sombre": (very_dark / total_pixels) * 100,
        "sombre": (dark / total_pixels) * 100,
        "moyen": (medium / total_pixels) * 100,
        "clair": (light / total_pixels) * 100,
        "très clair": (very_light / total_pixels) * 100
    }

    active = [cat for cat, pct in percentages.items() if pct > 10]

    commentaire_dominante = f"Votre œuvre présente une dominance de tons {dominant}."

    if len(active) <= 2:
        commentaire_variete = (
            "Elle semble présenter une gamme de valeurs limitée. "
            "Ce manque de contraste peut rendre la lecture visuelle plus uniforme, voire monotone."
        )
    elif len(active) in [3, 4]:
        commentaire_variete = (
            "Elle utilise une gamme de valeurs variée, ce qui crée un bon équilibre visuel "
            "et facilite la perception de la profondeur et du volume."
        )
    else:
        commentaire_variete = (
            "Elle exploite pleinement l'échelle des valeurs, ce qui renforce la structure visuelle "
            "et apporte une richesse de contrastes."
        )

    # valeurs extrêmes
    suggestions = []
    if percentages["très sombre"] < 5:
        suggestions.append("Des ombres plus intenses pourraient renforcer la profondeur et mieux structurer les volumes.")
    if percentages["très clair"] < 5:
        suggestions.append("L'ajout de lumières plus franches apporterait davantage de contraste et de relief à l’ensemble.")

    commentaire_suggestion = " ".join(suggestions)

    # image simplifiée par blocs de valeurs
    masses_image = np.zeros_like(gray_image)
    masses_image[(gray_image >= 0) & (gray_image < 52)] = 32
    masses_image[(gray_image >= 52) & (gray_image < 103)] = 80
    masses_image[(gray_image >= 103) & (gray_image < 154)] = 128
    masses_image[(gray_image >= 154) & (gray_image < 205)] = 176
    masses_image[(gray_image >= 205)] = 224

    masses_filename = f"masses_{original_filename}"
    masses_path = os.path.join("app", "static/images", masses_filename)
    cv2.imwrite(masses_path, masses_image)

    bar_values = [very_dark, dark, medium, light, very_light]
    bar_labels = ["Très sombre", "Sombre", "Moyen", "Clair", "Très clair"]

    plt.figure()
    plt.bar(bar_labels, bar_values, color='gray')
    plt.title("Répartition des valeurs")
    plt.xlabel("Tons")
    plt.ylabel("Nombre de pixels")
    plt.tight_layout()

    histogram_filename = f"hist_{original_filename}"
    histogram_path = os.path.join("app", "static/images", histogram_filename)
    plt.savefig(histogram_path)
    plt.close()

    # courbe de densité
    pixels = gray_image.flatten()

    plt.figure(figsize=(10, 5))

    kde = sns.kdeplot(pixels, bw_adjust=0.5)
    x_vals, y_vals = kde.get_lines()[0].get_data()
    plt.clf()

    slopes = np.abs(np.gradient(y_vals))
    slope_std = np.std(slopes)

    # Seuils (ajustables)
    low_thresh = 0.002
    mid_thresh = 0.007
    group_size = 5  # pour regrouper visuellement les zones

    if slope_std < low_thresh:
        commentaire_fluidite = (
            "Les transitions entre les tons sont très douces, "
            "ce qui renforce l’harmonie générale et donne à l’œuvre une ambiance feutrée ou enveloppante."
        )
    elif slope_std > mid_thresh:
        commentaire_fluidite = (
            "Les transitions tonales sont abruptes, avec des ruptures marquées entre les zones claires et sombres. "
            "Cela accentue le contraste et donne du caractère à la composition, mais peut aussi fragmenter la lecture."
        )
    else:
        commentaire_fluidite = (
            "Les transitions sont modérées et bien dosées, apportant à la fois lisibilité et rythme visuel."
        )

    commentaire_final = f"{commentaire_dominante} {commentaire_variete} {commentaire_suggestion} {commentaire_fluidite}"

    # Affichage des bandes
    for i in range(0, len(x_vals) - group_size, group_size):
        start = x_vals[i]
        end = x_vals[i + group_size]
        avg_slope = np.mean(slopes[i:i + group_size])

        if avg_slope < low_thresh:
            color = '#a8e6a3'  # vert vif, transition douce
        elif avg_slope < mid_thresh:
            color = '#ffcc80'  # orange clair, modéré
        else:
            color = '#ff9999'  # rouge rosé, rupture

        plt.axvspan(start, end, color=color, alpha=0.5)

    sns.kdeplot(pixels, bw_adjust=0.5, fill=True, color='black')

    plt.title(
        "Étude de la fluidité des transitions tonales\nZones colorées : stabilité ou rupture tonale locale",
        fontsize=13
    )
    plt.xlabel("Valeur de gris (0 = noir, 255 = blanc)")
    plt.ylabel("Densité")
    plt.tight_layout()

    legend_elements = [
        Patch(facecolor='#a8e6a3', edgecolor='black', label='fluide'),
        Patch(facecolor='#ffcc80', edgecolor='black', label='modérée'),
        Patch(facecolor='#ff9999', edgecolor='black', label='abrupte'),
    ]

    plt.legend(handles=legend_elements, title="Type de transition", loc='upper right')

    density_filename = f"density_{original_filename}"
    density_path = os.path.join("app", "static/images", density_filename)
    plt.savefig(density_path)
    plt.close()

    return commentaire_final, masses_filename, histogram_filename, density_filename