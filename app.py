import os
from moviepy.editor import VideoFileClip

# Define the folder path
folder_path = 'data/Bangla Folk/'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.mp4'):
        # Full path to the input video
        video_path = os.path.join(folder_path, filename)
        
        # Full path to the output audio
        audio_filename = filename.replace('.mp4', '.mp3')
        audio_path = os.path.join(folder_path, audio_filename)
        
        # Load the video file
        video = VideoFileClip(video_path)
        
        # Extract the audio
        audio = video.audio
        
        # Write the audio to MP3 file
        audio.write_audiofile(audio_path)
        
        # Close the video to free resources
        video.close()
        
        print(f"Converted {filename} to {audio_filename}")

print("All MP4 files have been converted to MP3.")
