import time
from threading import Thread, Lock
import sys

lock = Lock()

def animate_text(text, delay=0.1):
    with lock:
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

def sing_lyric(lyric, delay, speed):
    time.sleep(delay)
    animate_text(lyric, speed)

def sing_song():
    lyrics = [
        ("\n""Wake up in the morning", 0.07),
        ("Everything's alright", 0.06),
        ("At the end of the story", 0.06),
        ("You're holdin' me tight", 0.06),
        ("I don't need to worry", 0.07),
        ("am I out of my mind?", 0.06),
        ("And, oh, it's hard to see you""\n", 0.05),
        ("But I wish you were right here", 0.05),
        ("Oh, it's hard to leave you", 0.06),
        ("When I get you everywhere", 0.06),
        ("All this time I'm thinking", 0.05),
        ("I'm strong enough to sink it", 0.07),
        ("Oh, no, I don't need you", 0.06),
        ("But I miss you, come here", 0.06),
        

    ]
    
    delays = [0.3, 2.6, 4.6, 6.5, 8.3, 10.2, 12.0, 14.0, 16.3, 18.0, 20.5, 22.0, 24.5, 26.0]
    
    threads = []
    for i in range(len(lyrics)):
        lyric, speed = lyrics[i]
        t = Thread(target=sing_lyric, args=(lyric, delays[i], speed))
        threads.append(t)
        t.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    sing_song()