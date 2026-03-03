import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


## how does it work ??
        ## just grab with your mouse the object you want to track and voila 

# 1) Load SAM
def load_sam(weights_path, device='cpu'):
    sam = sam_model_registry["vit_b"](checkpoint=weights_path)
    sam.to(device)
    return SamPredictor(sam)

# 2) get mask from SAM
def sam_segment(predictor, frame, box):
    if frame.ndim == 2 or frame.shape[2] == 1:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    x, y, w, h = box
    box_np = np.array([x, y, x+w, y+h])
    masks, _, _ = predictor.predict(box=box_np)
    return (masks[0] * 255).astype(np.uint8)

# 3) compute ROI histogram
def compute_roi_hist(frame, mask, rect):
    x, y, w, h = rect
    roi = frame[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if np.std(hsv_roi[:,:,0]) < 5 and np.std(hsv_roi[:,:,1]) < 5:
        hist = cv2.calcHist([hsv_roi], [2], roi_mask, [256], [0,256])
    else:
        hist = cv2.calcHist([hsv_roi], [0,1], roi_mask, [180,256], [0,180,0,256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

# 4) Track using backprojection + meanshift --> (for object tracking)
def track_meanshift(video_path, rect, predictor):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    mask = sam_segment(predictor, frame, rect)
    roi_hist = compute_roi_hist(frame, mask, rect)
    track_window = rect
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if roi_hist.shape[0] == 256:
            Mask = cv2.calcBackProject([hsv], [2], roi_hist, [0,256], 1)
        else:
            Mask = cv2.calcBackProject([hsv], [0,1], roi_hist, [0,180,0,256], 1)
        _, track_window = cv2.meanShift(Mask, track_window, term_crit)
        x, y, w, h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("Mask",Mask)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 5) Process video
sam_weights = "sam_vit_b_01ec64.pth"

def process_video_1():
    cap = cv2.VideoCapture("Sample 1.mp4")
    ret, frame = cap.read()
    cap.release()
    rect = cv2.selectROI("Select Object - Video 1", frame, False, False)
    cv2.destroyAllWindows()
    predictor = load_sam(sam_weights)
    track_meanshift("Sample 1.mp4", rect, predictor)
    
def process_video_2():
    cap = cv2.VideoCapture("Sample 2.mp4")
    ret, frame = cap.read()
    cap.release()
    rect = cv2.selectROI("Select Object - Video 2", frame, False, False)
    cv2.destroyAllWindows()
    predictor = load_sam(sam_weights)
    track_meanshift("Sample 2.mp4", rect, predictor)

if __name__ == "__main__":
    process_video_1()
    process_video_2()

#----------------------------------------------------------------------------------------------------------------------------------------

### this code is for trying color tracking for segmentation


# class ColorObjectTracker:
#     def __init__(self, video_path, color_range, clahe_clip=0.01, clahe_grid=(4,4), morph_kernel=(5,5)):
#         self.cap = cv2.VideoCapture(video_path)
#         self.color_range = color_range
#         self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
#         self.kernel = np.ones(morph_kernel, np.uint8)
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
#         self.track_window = None
#         self.roi_hist = None
#         self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

#     def remove_lighting(self, frame):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)
#         v_eq = self.clahe.apply(v)
#         hsv_eq = cv2.merge([h, s, v_eq])
#         return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

#     def segment_color(self, frame, fgmask):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         color_mask = cv2.inRange(hsv, self.color_range[0], self.color_range[1])
#         moving_mask = cv2.bitwise_and(color_mask, color_mask, mask=fgmask)
#         moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_OPEN, self.kernel)
#         moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, self.kernel)

#         contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             return None, moving_mask

#         c = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(c)
#         return (x, y, w, h), moving_mask

#     def initialize_tracker(self, first_frame):
#         fgmask = self.bg_subtractor.apply(first_frame)
#         self.track_window, _ = self.segment_color(first_frame, fgmask)
#         if self.track_window is None:
#             print("Object not detected in first frame.")
#             return False

#         x, y, w, h = self.track_window
#         roi = first_frame[y:y+h, x:x+w]
#         hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         self.roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0,180,0,256])
#         cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
#         return True

#     def track(self):
#         ret, first_frame = self.cap.read()
#         if not ret:
#             print("Cannot open video.")
#             return

#         first_frame = self.remove_lighting(first_frame)
#         if not self.initialize_tracker(first_frame):
#             return

#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             frame_corr = self.remove_lighting(frame)
#             fgmask = self.bg_subtractor.apply(frame_corr)

#             hsv = cv2.cvtColor(frame_corr, cv2.COLOR_BGR2HSV)
#             backproj = cv2.calcBackProject([hsv], [0,1], self.roi_hist, [0,180,0,256], 1)
#             backproj = cv2.bitwise_and(backproj, backproj, mask=fgmask)

#             _, self.track_window = cv2.meanShift(backproj, self.track_window, self.term_crit)
#             x, y, w, h = self.track_window
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

#             cv2.imshow("Tracking", frame)
#             cv2.imshow("BackProj", backproj)

#             if cv2.waitKey(20) & 0xFF == 27:
#                 break
                
#         self.cap.release()
#         cv2.destroyAllWindows()


# blue_range = (np.array([100,170,70]), np.array([150,255,255]))
# black_range = (np.array([0,0,0]), np.array([180,255,50]))

# video1 = ColorObjectTracker("Sample 1.mp4", blue_range, morph_kernel=(4,4))
# video1.track()

# video2 = ColorObjectTracker("Sample 2.mp4", black_range)
# video2.track()
