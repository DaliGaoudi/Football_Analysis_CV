import cv2

def get_cropped_player_img(tracks, frames, image = False):
    if image:
        return

    for track_id, player in tracks["players"][0].items():
        frame = frames[0]
        bbox = player["bbox"]
        cropped_image = frame[int(bbox[1]) : int(bbox[3]),int( bbox[0]) : int(bbox[2])]
        cv2.imwrite("output_videos/cropped_img.jpg", cropped_image)
        image = True

        break