import numpy as np
import cv2
# import PIL
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd

#https://chev.me/arucogen/ avec les paramètres 4x4

#choix du dictionnaire
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

for i in range(4):
    # Identifiant du marqueur
    marker_id = i + 5
    # Taille de l'image à générer
    marker_size = 250
    # Génération d'une image du marqueur
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

    # Créer une nouvelle fenêtre pour chaque marqueur
    cv2.namedWindow(f'Marqueur {marker_id}', cv2.WINDOW_NORMAL)

    # Afficher l'image du marqueur dans la fenêtre correspondante
    cv2.imshow(f'Marqueur {marker_id}', marker_image)


cv2.waitKey(0)
cv2.destroyAllWindows() 