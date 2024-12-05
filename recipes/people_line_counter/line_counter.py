from typing import Optional

import cv2
import numpy as np

from trolo import DetectionPredictor, to_sv
import supervision as sv


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
         vis: Optional[bool]=True,) -> int:
    predictor = DetectionPredictor(model=model_name)
    video_info = sv.VideoInfo.from_video_path(video_path)

    # Change as per your requirement
    color_lookup = sv.ColorLookup.TRACK
    START = sv.Point(0, video_info.height // 2)
    END = sv.Point(video_info.width, video_info.height // 2)

    frames_generator = sv.get_video_frames_generator(video_path)
    if output_path:
        video_sink = sv.VideoSink(target_path=str(output_path), video_info=video_info).__enter__()

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(color_lookup=color_lookup)
    track_annotator = sv.TraceAnnotator(color_lookup=color_lookup)

    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4,
        text_thickness=4,
        text_scale=2)
    for frame in frames_generator:
        detected_objects = detect_objects(model_predictor=predictor, image=frame)
        tracked_detections = tracker.update_with_detections(detected_objects)
        line_zone.trigger(tracked_detections)

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
    main(video_path="people-walking.mp4", output_path="line_counter_output.mp4", vis=True)
