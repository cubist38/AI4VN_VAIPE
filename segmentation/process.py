import numpy as np
import statistics

def get_all_mask_points(mask):
    '''
    Output:
        tuple(
            x_points,
            y_points
        )
    '''
    indices = np.where(mask > 0)
    return indices

def get_rectangle_from_points(points):
    ymin, ymax = points[0].min(), points[0].max()
    xmin, xmax = points[1].min(), points[1].max()
    return xmin, ymin, xmax, ymax

def remove_inconsistent_mask(raw_masks, coeff_up = 1.5, coeff_down = 1.5):
    final_masks = []
    widths, heights = [], []
    for raw_mask in raw_masks:
        points = get_all_mask_points(raw_mask)
        xmin, ymin, xmax, ymax = get_rectangle_from_points(points)
        width, height = xmax - xmin, ymax - ymin
        widths.append(width)
        heights.append(height)
    
    width_var = statistics.mean(widths)
    width_std = statistics.stdev(widths)
    height_var = statistics.mean(heights)
    height_std = statistics.stdev(widths)

    for idx, (width, height) in enumerate(zip(widths, heights)):
        if width_var - coeff_down*width_std <= width <= width_var + coeff_up*width_std and \
            height_var - coeff_down*height_std <= height <= height_var + coeff_up*height_std:
            final_masks.append(raw_masks[idx])

    return np.array(final_masks)