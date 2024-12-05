from typing import Optional
import argparse

import cv2
import numpy as np
import supervision as sv
from trolo import DetectionPredictor, to_sv


def detect_objects(model_predictor: DetectionPredictor,
                   image: np.ndarray,
                   conf_threshold: Optional[float]=0.35) -> sv.Detections:
    """
    Detect objects in an image using the provided model predictor.
    """
    image = sv.cv2_to_pillow(image)
    results = model_predictor.predict([image], conf_threshold=conf_threshold)
    detections = to_sv(results[0])
    return detections


def main(video_path: str,
         model_name: Optional[str]="dfine-m" ,
         output_path: Optional[str]=None,
         vis: Optional[bool]=True,
         conf_threshold: Optional[float]=0.35) -> int:
    predictor = DetectionPredictor(model=model_name)
    video_info = sv.VideoInfo.from_video_path(video_path)

    # Change as per your requirement
    color_lookup = sv.ColorLookup.TRACK
    START = sv.Point(0, video_info.height // 2)
    END = sv.Point(video_info.width, video_info.height // 2)

    frames_generator = sv.get_video_frames_generator(video_path)
    if output_path:
        video_sink = sv.VideoSink(target_path=str(output_path), video_info=video_info).__enter__()

    # Initialize the tracker and annotators
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(color_lookup=color_lookup)
    track_annotator = sv.TraceAnnotator(color_lookup=color_lookup)

    # Initialize the line zone counter and annotator
    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4,
        text_thickness=4,
        text_scale=2)
    for frame in frames_generator:
        # Detect objects in the frame
        detected_objects = detect_objects(model_predictor=predictor, image=frame, conf_threshold=conf_threshold)
        # Update the tracker with the detected objects
        tracked_detections = tracker.update_with_detections(detected_objects)
        # Update the line zone counter
        line_zone.trigger(tracked_detections)

        # Annotate the frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, tracked_detections)
        labels = [
            f"{track_id[0]}"
            for track_id in zip(tracked_detections.tracker_id)
        ]
        annotated_frame = label_annotator.annotate(annotated_frame, tracked_detections, labels)
        annotated_frame = track_annotator.annotate(annotated_frame, tracked_detections)
        annotated_frame = line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)

        if vis:
            cv2.imshow("Annotated Frame", annotated_frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        if output_path:
            annotated_frame = sv.pillow_to_cv2(annotated_frame)
            video_sink.write_frame(annotated_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection and tracking in video with line zone counting.")
    parser.add_argument("--video_path", type=str, default="people-walking.mp4", help="Path to the input video file.")
    parser.add_argument("--model_name", type=str, default="dfine-m", help="Name of the detection model.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output annotated video.")
    parser.add_argument("--vis", type=bool, default=True,
                        help="Whether to visualize the output frames (default: True).")
    parser.add_argument("--conf_threshold", type=float, default=0.35,
                        help="Confidence threshold for detection (default: 0.35).")

    args = parser.parse_args()
    main(
        video_path=args.video_path,
        model_name=args.model_name,
        output_path=args.output_path,
        vis=args.vis,
        conf_threshold=float(args.conf_threshold)
    )
