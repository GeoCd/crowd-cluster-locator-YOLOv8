import numpy as np

from ultralytics import YOLO

import config

class CrowdDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_WEIGHTS)

    def detect(self, frame: np.ndarray):
        """Corre inferencia sobre el frame y retorna bounding boxes filtrados por clase persona, confidence e IOU."""
        """Runs inference on the frame and returns bounding boxes filtered by person class, confidence and IOU."""
        results = self.model(
                            frame,
                            conf=config.CONFIDENCE_THRESHOLD,
                            iou=config.IOU_THRESHOLD,
                            classes=[config.PERSON_CLASS_ID],
                            verbose=False
                            )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0,5))

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy().reshape(-1,1)
        return np.hstack([boxes, confs])

    def get_centroids(self, detectedBoxes: np.ndarray):
        """Extrae el punto central inferior de cada bounding box como posición espacial de la persona."""
        """Extracts the bottom center point of each bounding box as the person's spatial position."""
        if len(detectedBoxes) == 0:
            return np.empty((0,2))
        cx = (detectedBoxes[:,0] + detectedBoxes[:,2])/2
        cy =  detectedBoxes[:,3]
        return np.column_stack([cx,cy])

    def count_by_zone(self,centroids: np.ndarray,frameHeight: int,frameWidth: int):
        """Asigna cada centroide a una celda del grid y retorna la matriz de conteo."""
        """Assigns each centroid to a grid cell and returns the count matrix."""
        zoneCounts = np.zeros((config.GRID_ROWS,config.GRID_COLS),dtype=int)

        if len(centroids) == 0:
            return zoneCounts

        cellH = frameHeight/config.GRID_ROWS
        cellW = frameWidth/config.GRID_COLS

        for cx,cy in centroids:
            row = min(int(cy//cellH),config.GRID_ROWS-1)
            col = min(int(cx//cellW),config.GRID_COLS-1)
            zoneCounts[row,col]+=1

        return zoneCounts