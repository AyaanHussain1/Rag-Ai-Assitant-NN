import os 
import yt_dlp
def video_downloader_script(video_url):
    output_folder = "videos"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Videos Folder Created")

    ydl_opts = {
        'format': 'best', 
        'outtmpl': os.path.join(output_folder, '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        # If one video in the course is deleted or broken, skip it and keep downloading the rest
        'ignoreerrors': True,
    }
    try:
        print("Starting Download... Plz Wait...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"\n✅ Done! Your video has been downloaded directly into the '{output_folder}' folder.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
if __name__ == "__main__":
    link = input("Paste Your Youtube Video Link Here: ").strip()
    if link:
        video_downloader_script(link)
    else:
        print("Plz Provide Valid Url")