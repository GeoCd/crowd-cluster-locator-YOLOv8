import argparse
from pathlib import Path

import cv2

import config
from detector import CrowdDetector
from visualizer import annotate_frame

def process_image(imagePath: str, vizMode: str, save: bool):
    """Lee una imagen, corre el pipeline completo y muestra o guarda el resultado."""
    """Reads an image, runs the full pipeline and displays or saves the result."""
    frame = cv2.imread(imagePath)
    if frame is None:
        print(f"Could not read: {imagePath}")
        return

    detector = CrowdDetector()
    detectedBoxes = detector.detect(frame)
    centroids     = detector.get_centroids(detectedBoxes)
    zoneCounts    = detector.count_by_zone(centroids, frame.shape[0], frame.shape[1])

    print(f"Detected persons: {len(detectedBoxes)}")
    print(f"Count per zone:\n{zoneCounts}")

    output = annotate_frame(frame, detectedBoxes, zoneCounts, vizMode)

    if save:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        outPath = config.RESULTS_DIR / Path(imagePath).name
        cv2.imwrite(str(outPath), output)
        print(f"Saved in: {outPath}")

    cv2.imshow("CrowdDetector", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(videoSource, vizMode: str, save: bool):
    """Procesa frame a frame un video o webcam, con opción de guardar el output como mp4."""
    """Processes a video or webcam frame by frame, with option to save the output as mp4."""
    cap = cv2.VideoCapture(videoSource)
    if not cap.isOpened():
        print(f"Could not open: {videoSource}")

        return

    detector  = CrowdDetector()
    videoWriter = None

    while True:
        ret, frame = cap.read()
        if not ret:

            break

        detectedBoxes = detector.detect(frame)
        centroids = detector.get_centroids(detectedBoxes)
        zoneCounts = detector.count_by_zone(centroids, frame.shape[0], frame.shape[1])
        output = annotate_frame(frame, detectedBoxes, zoneCounts, vizMode)

        if save and videoWriter is None:
            config.RESULTS_DIR.mkdir(exist_ok=True)
            sourceName = "webcam" if videoSource == 0 else Path(str(videoSource)).stem
            outputPath = config.RESULTS_DIR / f"{sourceName}_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            videoWriter = cv2.VideoWriter(str(outputPath), fourcc, 30, (frame.shape[1], frame.shape[0]))

        if videoWriter:
            videoWriter.write(output)

        cv2.imshow("CrowdDetector", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()
    if videoWriter:
        videoWriter.release()

    cv2.destroyAllWindows()

def main():
    """Parsea los argumentos de línea de comandos y enruta al procesador correspondiente."""
    """Parses command line arguments and routes to the corresponding processor."""
    argparser = argparse.ArgumentParser(description="CrowdDetector - YOLOv8 + density analysis")
    argparser.add_argument("--source", type=str, default=None)
    argparser.add_argument("--mode", type=str, default=config.VIZ_MODE,choices=["heatmap", "grid", "both"])
    argparser.add_argument("--save", action="store_true")
    args = argparser.parse_args()

    if args.source is None or args.source == "webcam":
        process_video(0, args.mode, args.save)
    elif args.source.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(args.source, args.mode, args.save)
    else:
        process_video(args.source, args.mode, args.save)

if __name__ == "__main__":
    main()
