from flask import Blueprint, render_template, request
import cv2
import os
from .analysis.valeurs import analyse_valeurs
from .analysis.composition import analyse_composition
from .analysis.couleurs import analyse_couleurs

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template("index.html")

@main.route('/valeurs', methods=['GET', 'POST'])
def valeurs():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return render_template("valeurs.html", analysis_result="Fichier non valide.")

        filepath = os.path.join("app", "static/images", file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        commentaire, hist, categories, density = analyse_valeurs(gray, file.filename)

        return render_template("valeurs.html",
            masses_filename="masses_" + file.filename,
            histogram_filename="hist_" + file.filename,
            density_filename="density_" + file.filename,
            analysis_result=commentaire
        )
    return render_template("valeurs.html")

@main.route('/composition', methods=['GET', 'POST'])
def composition():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return render_template("composition.html", resultats=[], submitted=True)

        filepath = os.path.join("app", "static/images", file.filename)
        file.save(filepath)

        resultats = analyse_composition(file.filename)

        return render_template("composition.html", resultats=resultats, submitted=True)

    return render_template("composition.html", resultats=[], submitted=False)

@main.route('/couleurs', methods=['GET', 'POST'])
def couleurs():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return render_template("couleurs.html", analysis_result="Fichier non valide.")

        filepath = os.path.join("app", "static/images", file.filename)
        file.save(filepath)

        resultats = analyse_couleurs(file.filename)

        return render_template("couleurs.html",
            resultats=resultats,
            submitted=True
        )
    return render_template("couleurs.html", resultats=[], submitted=False)