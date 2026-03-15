import cv2
import numpy as np

import config

def gaussian_heatmap(centroids: np.ndarray, frameHeight: int, frameWidth: int):
    """Genera un mapa de calor acumulando distribuciones gaussianas centradas en cada detección."""
    """Generates a heat map by accumulating gaussian distributions centered on each detection."""
    heatmap = np.zeros((frameHeight, frameWidth), dtype=np.float32)
    radius  = config.HEATMAP_RADIUS

    for cx, cy in centroids:
        x1 = max(0, int(cx) - radius * 3)
        x2 = min(frameWidth,  int(cx) + radius * 3)
        y1 = max(0, int(cy) - radius * 3)
        y2 = min(frameHeight, int(cy) + radius * 3)

        for px in range(x1, x2):
            for py in range(y1, y2):
                dist2 = (px - cx) ** 2 + (py - cy) ** 2
                heatmap[py, px] += np.exp(-dist2 / (2 * radius ** 2))

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def draw_heatmap(frame: np.ndarray, centroids: np.ndarray):
    """Convierte el heatmap a colormap y lo mezcla sobre el frame original."""
    """Converts the heatmap to JET colormap and blends it over the original frame."""
    frameH, frameW = frame.shape[:2]
    heatmap = gaussian_heatmap(centroids, frameH, frameW)

    heatmapU8 = (heatmap * 255).astype(np.uint8)
    heatmapRGB = cv2.applyColorMap(heatmapU8, cv2.COLORMAP_JET)

    output = cv2.addWeighted(frame, 1 - config.HEATMAP_ALPHA, heatmapRGB, config.HEATMAP_ALPHA, 0)

    return output

def draw_grid(frame: np.ndarray, zoneCounts: np.ndarray):
    """Dibuja el grid de zonas con intensidad de color proporcional a la densidad y borde rojo en zonas de alerta."""
    """Draws the zone grid with color intensity proportional to density and red border on alert zones."""
    output = frame.copy()
    frameH, frameW = frame.shape[:2]
    cellH = frameH / config.GRID_ROWS
    cellW = frameW / config.GRID_COLS
    if zoneCounts.max() > 0:
        maxCount = zoneCounts.max() 
    else:
        maxCount = 1

    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            count = zoneCounts[row, col]
            x1 = int(col * cellW)
            y1 = int(row * cellH)
            x2 = int(x1 + cellW)
            y2 = int(y1 + cellH)

            intensity = int(200 * count / maxCount)
            alertColor = (0, 0, intensity)

            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), alertColor, -1)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

            borderColor = (0, 0, 255) if count >= config.DENSITY_ALERT_THRESHOLD else (200, 200, 200)
            cv2.rectangle(output, (x1, y1), (x2, y2), borderColor, 2)

            if count > 0:
                label = str(count)
            else:
                label = ""

            if label:
                cv2.putText(output, label, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return output

def draw_detections(frame: np.ndarray, detectedBoxes: np.ndarray):
    """Dibuja los bounding boxes con su confidence score sobre el frame."""
    """Draws bounding boxes with their confidence score over the frame."""
    output = frame.copy()
    for box in detectedBoxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output

def annotate_frame(frame: np.ndarray, detectedBoxes: np.ndarray, zoneCounts: np.ndarray, vizualizationMode: str):
    """Orquesta todas las visualizaciones según el modo configurado y agrega el conteo total."""
    """Orchestrates all visualizations according to the configured mode and adds the total count."""
    output = draw_detections(frame, detectedBoxes)

    if vizualizationMode in ("heatmap", "both"):
        if len(detectedBoxes) > 0:
            centroids = np.column_stack([(detectedBoxes[:, 0] + detectedBoxes[:, 2]) / 2, detectedBoxes[:, 3]])
        else:
            centroids = np.empty((0, 2))

        output = draw_heatmap(output, centroids)

    if vizualizationMode in ("grid", "both"):
        output = draw_grid(output, zoneCounts)

    totalPersons = len(detectedBoxes)
    cv2.putText(output, f"Total Persons: {totalPersons}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    return output
