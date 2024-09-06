from utils import read_video, save_video, get_cropped_player_img
from trackers import Tracker 
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
import numpy as np

def main():
    frames = read_video("assets/game.mp4")

    ##init tracker 
    tracker = Tracker("Yolo_Training/models/best.pt")

    tracks = tracker.get_object_tracks(frames,
                                       read_from_stub = True,
                                       stub_path = "stubs/track_stubs.pkl")   
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    get_cropped_player_img(tracks, frames, image=False)

    assigner =  TeamAssigner()
    assigner.assign_team_color(frames[0],
                               tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = assigner.assign_player(frames[frame_num],
                                              track["bbox"],
                                              player_id)
            
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assignerd_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)   

        if assignerd_player != -1:
            tracks["players"][frame_num][assignerd_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assignerd_player]["team"])
        else: 
            team_ball_control.append(team_ball_control[-1])    
    team_ball_control = np.array(team_ball_control)

    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control) 

    save_video(output_frames,"output_videos/video.avi")
 


main()   