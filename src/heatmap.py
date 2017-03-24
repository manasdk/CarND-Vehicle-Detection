import numpy as np


class Heatmapper:

    def __init__(self, img):
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

    def add_heat(self, bbox_list, threshold):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return heatmap
        return self._apply_threshold(threshold)

    def _apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= threshold] = 0
        # Return thresholded map
        return self.heatmap
