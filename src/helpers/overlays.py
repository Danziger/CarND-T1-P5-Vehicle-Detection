def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    """
    Returns a copy of the provided image with boxes (rectangles)
    drawn on it using the specified color and thickness.
    
    """

    img = np.copy(img)
        
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thick)
    
    return img
