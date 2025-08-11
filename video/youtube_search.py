import re
import requests
from youtubesearchpython import VideosSearch

def search_youtube_videos(query, limit=7):
    videosSearch = VideosSearch(query, limit=limit)
    results = videosSearch.result().get("result", [])
    videos = []
    for v in results:
        video_info = {
            "title": v.get("title"),
            "url": v.get("link"),
            "thumbnail": v.get("thumbnails")[0]["url"],
            "id": v.get("id"),
        }
        videos.append(video_info)
    return videos

def youtube_title(video_id):
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        resp = requests.get(url, timeout=1)
        resp.raise_for_status()
        return resp.json().get("title")
    
    except Exception as e:
        print("Error fetching title:", e)
        return None


def youtube_id(url_or_text: str) -> str | None:
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url_or_text)
    
    if match:
        id = match.group(1)
        return id, youtube_title(id)
    else: None