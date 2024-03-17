import cv2
import numpy as np
import math


def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    sort_coords = list_of_xy_coords[indices]
    sort_coords = np.roll(sort_coords, -1, axis=0)
    return sort_coords


image = cv2.imread('test-aruco4.jpg')
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


    # appel de la fonction pour trier les coordonnées selon le sens horaire
    croppedCorners = sort_coordinates(np.array(croppedCorners, dtype=np.float32))

    print(croppedCorners)

    # Coordonnées du rectangle de destination
    coordonnees = croppedCorners
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
