"""
Download the karate YouTube video for SAM 3D Body processing.
URL: https://www.youtube.com/watch?v=1MrRmimBJoA

Usage:
    python download_karate_video.py

Output:
    ./input_video/karate_video.mp4   (original)
    ./input_video/karate_video_720p.mp4  (optional 720p copy)
"""

import os
import subprocess
import sys

VIDEO_URL = "https://www.youtube.com/watch?v=1MrRmimBJoA"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "input_video")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "karate_video.mp4")


def check_yt_dlp():
    """Make sure yt-dlp is installed."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        print(f"[✔] yt-dlp version: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[✘] yt-dlp not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp", "--upgrade"], check=True)
        return True


def check_ffmpeg():
    """Make sure ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.split("\n")[0]
        print(f"[✔] {first_line}")
        return True
    except FileNotFoundError:
        print("[!] ffmpeg not found in PATH.")
        print("    On Windows: winget install ffmpeg  OR  choco install ffmpeg")
        print("    Or download from https://ffmpeg.org/download.html")
        return False


def get_cookie_args() -> list:
    """
    Determine cookie strategy for YouTube authentication.
    Priority:
      1. cookies.txt file in current directory (most reliable)
      2. Browser cookies (edge, chrome, etc.)
      3. No cookies (will likely fail for bot-detected videos)

    To create cookies.txt:
      - Install 'Get cookies.txt LOCALLY' browser extension
      - Visit youtube.com while signed in
      - Click the extension and export for current site
      - Place cookies.txt next to this script
    """
    # 1. Check for cookies.txt file
    cookies_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    if os.path.exists(cookies_file):
        print(f"[OK] Using cookies.txt: {cookies_file}")
        return ["--cookies", cookies_file]

    # 2. Try browser cookies
    for browser in ["edge", "chrome", "firefox", "opera", "brave"]:
        try:
            result = subprocess.run(
                ["yt-dlp", "--cookies-from-browser", browser, "--no-download",
                 "--print", "%(title)s", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                capture_output=True, text=True, timeout=25
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"[OK] Using cookies from browser: {browser}")
                return ["--cookies-from-browser", browser]
        except Exception:
            continue

    print("[!] No authenticated browser session detected.")
    print()
    print("    *** HOW TO FIX YOUTUBE BOT DETECTION ***")
    print("    =========================================")
    print("    OPTION A (RECOMMENDED) - Export cookies.txt:")
    print("      1. Install Chrome/Edge extension: 'Get cookies.txt LOCALLY'")
    print("      2. Go to https://youtube.com while signed in")
    print("      3. Click extension -> Export -> For current site")
    print("      4. Save as 'cookies.txt' in this folder:")
    print(f"         {os.path.dirname(os.path.abspath(__file__))}")
    print("      5. Re-run: python download_karate_video.py")
    print()
    print("    OPTION B - Sign into YouTube in Edge/Chrome, close the browser,")
    print("      then re-run this script.")
    print()
    print("    OPTION C - Download manually via browser and save to:")
    print(f"      {OUTPUT_PATH}")
    print("    =========================================")
    print()
    return []


def get_video_info(url: str, cookie_args: list = None):
    """Fetch and display video metadata before downloading."""
    print(f"\n[→] Fetching video info for: {url}")
    cmd = ["yt-dlp"] + (cookie_args or []) + [
        "--no-download",
        "--print", "%(title)s",
        "--print", "%(duration_string)s",
        "--print", "%(upload_date)s",
        "--print", "%(uploader)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    lines = result.stdout.strip().split("\n")
    print(f"\n{'='*55}")
    print(f"  Title    : {lines[0] if len(lines) > 0 else 'N/A'}")
    print(f"  Duration : {lines[1] if len(lines) > 1 else 'N/A'}")
    print(f"  Uploaded : {lines[2] if len(lines) > 2 else 'N/A'}")
    print(f"  Channel  : {lines[3] if len(lines) > 3 else 'N/A'}")
    print(f"{'='*55}\n")


def list_formats(url: str, cookie_args: list = None):
    """List all available formats/qualities."""
    print("[→] Available video formats:")
    subprocess.run(
        ["yt-dlp"] + (cookie_args or []) + ["-F", url],
        timeout=60
    )


def download_video(url: str, output_path: str, cookie_args: list = None):
    """
    Download the best quality mp4 video.
    Strategy: best video (up to 1080p) + best audio merged into mp4.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"[✔] Video already downloaded: {output_path}")
        probe_video(output_path)
        return output_path

    print(f"[→] Downloading video to: {output_path}")
    print("    Format: best mp4 up to 1080p (ideal for 4090 processing)")

    cmd = (
        ["yt-dlp"]
        + (cookie_args or [])
        + [
            # Best video up to 1080p merged with best audio
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", output_path,
            "--write-thumbnail",
            "--no-playlist",
            "--progress",
            "--newline",
            url,
        ]
    )

    print(f"\n[→] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, timeout=600)

    if result.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n[✔] Download complete!")
        print(f"    File: {output_path}")
        print(f"    Size: {size_mb:.1f} MB")
        return output_path
    else:
        print(f"\n[✘] Download failed with code {result.returncode}")
        # Try a more permissive format as fallback
        print("[→] Trying fallback format (best available)...")
        cmd_fallback = (
            ["yt-dlp"]
            + (cookie_args or [])
            + ["-f", "best", "--merge-output-format", "mp4",
               "--output", output_path, "--progress", "--newline", url]
        )
        result2 = subprocess.run(cmd_fallback, timeout=600)
        if result2.returncode == 0:
            print(f"[✔] Fallback download succeeded: {output_path}")
            return output_path
        raise RuntimeError("Video download failed")


def probe_video(video_path: str):
    """
    Use ffprobe to get precise video info: resolution, fps, duration.
    Returns a dict with video statistics.
    """
    print(f"\n[→] Probing video: {video_path}")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                video_path,
            ],
            capture_output=True, text=True, timeout=30, check=True
        )
        import json
        info = json.loads(result.stdout)

        video_stream = None
        audio_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_stream = stream

        fmt = info.get("format", {})

        stats = {}
        if video_stream:
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            codec = video_stream.get("codec_name", "unknown")
            # fps as fraction
            fps_str = video_stream.get("r_frame_rate", "0/1")
            num, den = [int(x) for x in fps_str.split("/")]
            fps = num / den if den != 0 else 0
            nb_frames = video_stream.get("nb_frames", "unknown")

            stats["width"] = width
            stats["height"] = height
            stats["fps"] = fps
            stats["codec"] = codec
            stats["nb_frames"] = nb_frames

        duration = float(fmt.get("duration", 0))
        size_bytes = int(fmt.get("size", 0))
        stats["duration_sec"] = duration
        stats["duration_str"] = f"{int(duration//60)}m {int(duration%60)}s"
        stats["size_mb"] = size_bytes / (1024 * 1024)
        stats["total_frames"] = int(fps * duration) if fps > 0 else 0

        print(f"\n{'='*55}")
        print(f"  VIDEO STATISTICS")
        print(f"{'='*55}")
        print(f"  Resolution  : {stats.get('width', '?')}x{stats.get('height', '?')} px")
        print(f"  FPS         : {stats.get('fps', 0):.3f}")
        print(f"  Duration    : {stats.get('duration_str', '?')} ({duration:.1f}s)")
        print(f"  Total Frames: {stats.get('total_frames', '?')}")
        print(f"  Video Codec : {stats.get('codec', '?')}")
        print(f"  File Size   : {stats.get('size_mb', 0):.1f} MB")
        if audio_stream:
            print(f"  Audio       : {audio_stream.get('codec_name', '?')} @ {audio_stream.get('sample_rate', '?')} Hz")
        print(f"{'='*55}\n")

        # Processing time estimate for RTX 4090
        total_frames = stats.get("total_frames", 0)
        if total_frames > 0:
            # Rough estimates based on model size and hardware
            # DINOv3-H+ on 4090: ~0.8-1.5s per frame (includes detector + SAM2 + MoGe2)
            sec_per_frame_fast = 0.8   # body-only mode
            sec_per_frame_full = 1.5   # full mode with hand decoder
            print(f"  PROCESSING TIME ESTIMATE (RTX 4090)")
            print(f"{'='*55}")
            print(f"  Body-only mode  : ~{total_frames * sec_per_frame_fast / 60:.0f} min  ({total_frames} frames × {sec_per_frame_fast}s)")
            print(f"  Full mode       : ~{total_frames * sec_per_frame_full / 60:.0f} min  ({total_frames} frames × {sec_per_frame_full}s)")
            print(f"  Tip: Use --frame_skip to process every Nth frame for speed")
            print(f"{'='*55}\n")

        return stats

    except FileNotFoundError:
        print("[!] ffprobe not found. Install ffmpeg to get video stats.")
        return {}
    except Exception as e:
        print(f"[!] Error probing video: {e}")
        return {}


def create_frame_sample(video_path: str, output_dir: str, n_frames: int = 5):
    """Extract a few sample frames for quick inspection."""
    sample_dir = os.path.join(output_dir, "sample_frames")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"[→] Extracting {n_frames} sample frames to: {sample_dir}")
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-print_format", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=10
    )
    try:
        duration = float(probe.stdout.strip())
    except:
        duration = 30.0  # fallback

    for i in range(n_frames):
        t = duration * (i + 1) / (n_frames + 1)
        out_frame = os.path.join(sample_dir, f"sample_{i+1:02d}_t{t:.1f}s.jpg")
        subprocess.run([
            "ffmpeg", "-ss", str(t), "-i", video_path,
            "-frames:v", "1", "-q:v", "2",
            out_frame, "-y", "-loglevel", "quiet"
        ], timeout=30)
        if os.path.exists(out_frame):
            print(f"    [✔] Frame @ {t:.1f}s → {os.path.basename(out_frame)}")

    print(f"[✔] Sample frames saved to: {sample_dir}")
    return sample_dir


def main():
    print("\n" + "="*55)
    print("  SAM 3D Body - Karate Video Downloader")
    print("="*55)
    print(f"  URL: {VIDEO_URL}")
    print("="*55 + "\n")

    # Step 1: Check tools
    print("[STEP 1] Checking required tools...")
    check_yt_dlp()
    ffmpeg_ok = check_ffmpeg()

    # Step 1b: Detect browser cookies for YouTube auth
    print("\n[STEP 1b] Detecting browser cookies for YouTube...")
    cookie_args = get_cookie_args()

    # Step 2: Get video info
    print("\n[STEP 2] Fetching video metadata...")
    try:
        get_video_info(VIDEO_URL, cookie_args)
    except Exception as e:
        print(f"[!] Could not fetch info (may still download): {e}")

    # Step 3: Optional - list formats
    try:
        import sys
        if "--list-formats" in sys.argv or "-F" in sys.argv:
            list_formats(VIDEO_URL, cookie_args)
            return
    except:
        pass

    # Step 4: Download
    print("[STEP 3] Downloading video...")
    try:
        video_path = download_video(VIDEO_URL, OUTPUT_PATH, cookie_args)
    except RuntimeError as e:
        print(f"[✘] {e}")
        sys.exit(1)

    # Step 5: Probe
    if ffmpeg_ok:
        print("[STEP 4] Probing video statistics...")
        stats = probe_video(video_path)

        # Step 6: Extract sample frames
        print("[STEP 5] Extracting sample frames for inspection...")
        sample_dir = create_frame_sample(video_path, OUTPUT_DIR, n_frames=8)

    print("\n" + "="*55)
    print("  NEXT STEPS:")
    print("="*55)
    print("  1. Review sample frames in:  input_video/sample_frames/")
    print("  2. Download model checkpoints:")
    print("       python download_models.py")
    print("  3. Run the full video pipeline:")
    print("       python process_karate_video.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
