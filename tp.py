import cv2
import numpy as np
import math

image = cv2.imread('test-aruco3.jpg')
# image = cv2.imread('flecheAR.jpg')
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

if len(markerCorners) == 4:

    # Affiche les informations liées aux aruco
    print(markerIds)
    # print(markerCorners)

    ids = markerIds.flatten()

    croppedCorners = []

    # boucle pour chaque aruco détecté
    for (markerCorner, markerID) in zip(markerCorners, ids):
        # extraire les angles des aruco (toujours dans l'ordre
        # haut-gauche, haut-droite, bas-gauche, bas-droit)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convertir en entier (pour l'affichage)
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # dessiner un quadrilatère autour de chaque aruco
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # calculer puis afficher un point rouge au centre
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # afficher l'identifiant
        cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # enregistre les coordonnées du centre des aruco dans un tableau
        croppedCorners.append((cX, cY))

    # tri des coordonnées pour former un quadrilatère dont les cotés ne se croisent pas
    # Trouver le coin supérieur gauche (le point avec les coordonnées minimales)
    coin_superieur_gauche = min(croppedCorners, key=lambda p: (p[0], p[1]))

    # Trier les coordonnées en fonction des angles polaires par rapport au coin supérieur gauche
    croppedCorners = sorted(croppedCorners, key=lambda p: math.atan2(p[1] - coin_superieur_gauche[1],
                                                                     p[0] - coin_superieur_gauche[0]))

    print(croppedCorners)

    # Coordonnées du rectangle de destination
    coordonnees = np.array(croppedCorners, dtype=np.float32)
    dimensions = (400, 300)
    coordonnees_dest = np.array(
        [[0, 0], [dimensions[0], 0], [dimensions[0], dimensions[1]], [0, dimensions[1]]],
        dtype=np.float32)

    # Calculer la matrice de transformation perspective
    matrix = cv2.getPerspectiveTransform(coordonnees, coordonnees_dest)

    # Appliquer la transformation perspective
    image_transformee = cv2.warpPerspective(image, matrix, dimensions)

    # Afficher l'image transformée
    cv2.imshow("Image Transformée", image_transformee)
    cv2.waitKey(0)


else:
    print(f"Erreur, nombre d'Aruco trouvé(s) : {len(markerCorners)}")
