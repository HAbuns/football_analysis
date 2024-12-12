from ultis import read_video, save_video
from tracker import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # read video
    video_frame = read_video(r'D:\Learn AI\football-analysis\input_video\08fd33_4.mp4')
    
    # Initialize tracker
    tracker = Tracker('model/best.pt')
    
    tracks = tracker.get_object_tracks(video_frame,
                                      read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    
    # interpolite ball position
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # assign team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frame[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frame[frame_num], track['bbox'], player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # save cropped image of a player
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frame[0]
        
        cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        cv2.imwrite(f'output_video/cropped_image.jpg', cropped_img)
        break
    
    # assign ball position
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
            
    # draw output
    ## draw object track
    output_video_frame = tracker.draw_annotations(video_frame, tracks, team_ball_control)
    
    # save video
    save_video(output_video_frame, r'D:\Learn AI\football-analysis\output_video\output_video.avi')

if __name__ == '__main__':
    main()
