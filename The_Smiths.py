import sys
import time

def animate_text(text, duration):
    delay = duration / len(text)

    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)

    print()

def sing_song():
    lyrics = [
        ("And if a double-decker bus", 0.00, 3.70),
        ("Crashes into us", 3.85, 2.10),
        ("To die by your side", 6.10, 2.60),
        ("Is such a heavenly way to die", 8.90, 4.20),

        ("And if a ten-ton truck", 13.40, 3.40),
        ("Kills the both of us", 16.95, 2.30),
        ("To die by your side", 19.45, 2.60),
        ("Well, the pleasure — the privilege is mine", 22.25, 4.80),
    ]

    start = time.perf_counter()

    for text, start_time, duration in lyrics:
        while time.perf_counter() - start < start_time:
            time.sleep(0.005)

        animate_text(text, duration)

sing_song()