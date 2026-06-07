import json
import os
import sys
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

mp_hands = mp.solutions.hands

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MSASL_TRAIN = os.path.join(DATA_DIR, 'MSASL_train.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'msasl_greetings.json')

TMP_CLIPS = os.path.join(DATA_DIR, '_msasl_clips')
os.makedirs(TMP_CLIPS, exist_ok=True)

GREETINGS = [
    'hello', 'bye', 'please', 'sorry', 'yes', 'no', 'good', 'nice',
    'happy', 'help', 'friend', 'love', 'water', 'eat', 'drink',
    'home', 'school', 'like', 'name', 'fine', 'how', 'what', 'where',
]
SAMPLES_PER_WORD = 10

_hands_obj = None
def get_hands():
    global _hands_obj
    if _hands_obj is None:
        _hands_obj = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
    return _hands_obj

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = get_hands().process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    flat = []
    for landmark in lm.landmark:
        flat.extend([landmark.x, landmark.y, landmark.z])
    return flat

def url_works(url):
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'yt_dlp', '-s', '--no-warnings', url],
            capture_output=True, timeout=15
        )
        return r.returncode == 0
    except:
        return False

def download_full_video(url, out_path):
    cmd = [
        sys.executable, '-m', 'yt_dlp', '-q',
        '-f', 'worstvideo[height<=360]',
        '-o', out_path, url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
    except:
        return False

def process_sample(sample):
    url = sample['url']
    start = sample['start_time']
    end = sample['end_time']
    box = sample.get('box')
    label = sample['clean_text'].upper()

    clip = os.path.join(TMP_CLIPS, f'clip_{abs(hash(url))}_{start}.mp4')
    if not download_full_video(url, clip):
        return None

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    max_frames = int((end - start) * fps)
    all_lm = []
    frames_read = 0
    while frames_read < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        if box and len(box) == 4:
            h, w = frame.shape[:2]
            x, y, bw, bh = [int(v) for v in (box[0]*w, box[1]*h, box[2]*w, box[3]*h)]
            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)
            if bw > 10 and bh > 10:
                frame = frame[y:y+bh, x:x+bw]
        lm = extract_landmarks(frame)
        if lm:
            all_lm.append(lm)
    cap.release()
    try:
        os.remove(clip)
    except:
        pass

    if len(all_lm) < 3:
        return None
    avg = np.mean(all_lm, axis=0).tolist()
    return {'label': label, 'numHands': 1, 'features': avg}

def main():
    with open(MSASL_TRAIN) as f:
        train = json.load(f)

    existing = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
    have = set(e['label'] for e in existing)
    results = list(existing)

    for word in GREETINGS:
        wl = word.upper()
        if wl in have:
            print(f'{wl}: already has {sum(1 for e in existing if e["label"] == wl)} samples, skipping')
            continue

        samples = [s for s in train if s['clean_text'] == word]
        working = []
        for s in samples:
            if url_works(s['url']):
                working.append(s)
            if len(working) >= SAMPLES_PER_WORD:
                break

        print(f'{wl}: {len(working)}/{len(samples)} URLs working, processing {min(SAMPLES_PER_WORD, len(working))}...', end=' ')
        sys.stdout.flush()

        count = 0
        for s in working[:SAMPLES_PER_WORD]:
            r = process_sample(s)
            if r:
                results.append(r)
                count += 1

        # Save after each word so partial progress isn't lost
        with open(OUTPUT_PATH + '.tmp', 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(OUTPUT_PATH + '.tmp', OUTPUT_PATH)

        print(f'{count} OK')
        have.add(wl)
        sys.stdout.flush()

    final = Counter(r['label'] for r in results)
    print(f'\nTotal: {len(results)} samples')
    for g in GREETINGS:
        c = final.get(g.upper(), 0)
        if c:
            print(f'  {g.upper()}: {c}')

    if _hands_obj:
        _hands_obj.close()

if __name__ == '__main__':
    main()
