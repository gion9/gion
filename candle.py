from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


_SIZE = 256
_MASK = _SIZE - 1
_RNG = np.random.default_rng(20240611)
_TABLE_2D = _RNG.random((_SIZE, _SIZE), dtype=np.float32)
_TABLE_1D = _RNG.random(_SIZE, dtype=np.float32)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def noise1(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    i = np.floor(x)
    f = _fade(x - i)
    i = i.astype(np.int32) & _MASK
    return _TABLE_1D[i] * (1.0 - f) + _TABLE_1D[(i + 1) & _MASK] * f


def nz(t: float, seed: float = 0.0) -> float:
    return float(noise1(t + seed * 37.0))


def noise2(x, y) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xi = np.floor(x)
    yi = np.floor(y)
    fx = _fade(x - xi)
    fy = _fade(y - yi)
    xi = xi.astype(np.int32) & _MASK
    yi = yi.astype(np.int32) & _MASK
    x1 = (xi + 1) & _MASK
    y1 = (yi + 1) & _MASK
    a = _TABLE_2D[yi, xi]
    b = _TABLE_2D[yi, x1]
    c = _TABLE_2D[y1, xi]
    d = _TABLE_2D[y1, x1]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def fbm2(x, y, octaves: int = 3, lacunarity: float = 2.03, gain: float = 0.5) -> np.ndarray:
    total = 0.0
    amp = 1.0
    norm = 0.0
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    for o in range(octaves):
        total = total + amp * noise2(x + 17.3 * o, y + 9.1 * o)
        norm += amp
        amp *= gain
        x = x * lacunarity
        y = y * lacunarity
    return total / norm


def smoothstep(a: float, b: float, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NR = len(RAMP) - 1
ALL_CHARS = RAMP + "="
CHAR_INDEX = {ch: i for i, ch in enumerate(ALL_CHARS)}

_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_FONT_CANDIDATES = ("consolab.ttf", "consola.ttf", "CascadiaMono.ttf", "lucon.ttf", "cour.ttf")

SOFTNESS = 0.42
AURA = 0.08


def find_font() -> str | None:
    for name in _FONT_CANDIDATES:
        path = os.path.join(_FONT_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="constant")
    h = (p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]) / 4.0
    return (h[:-2, :] + 2.0 * h[1:-1, :] + h[2:, :]) / 4.0


def build_atlas(cell_w: int, cell_h: int) -> np.ndarray:
    pygame.font.init()
    path = find_font()
    atlas = np.zeros((len(ALL_CHARS), cell_h, cell_w), dtype=np.float32)

    size = max(6, int(cell_h * 0.95))
    while size > 6:
        font = pygame.font.Font(path, size) if path else pygame.font.SysFont("monospace", size)
        if font.size("M")[0] <= cell_w and font.get_ascent() + font.get_descent() <= cell_h * 1.25:
            break
        size -= 1
    else:
        font = pygame.font.Font(path, size) if path else pygame.font.SysFont("monospace", size)

    baseline = (cell_h + font.get_ascent() - abs(font.get_descent())) // 2
    for i, ch in enumerate(ALL_CHARS):
        if ch == " ":
            continue
        surf = font.render(ch, True, (255, 255, 255), (0, 0, 0))
        glyph = pygame.surfarray.array3d(surf)[:, :, 0].T
        gh, gw = glyph.shape
        ox = max(0, (cell_w - gw) // 2)
        oy = baseline - font.get_ascent()
        y0 = max(0, oy)
        y1 = min(cell_h, oy + gh)
        x1 = min(cell_w, ox + gw)
        atlas[i, y0:y1, ox:x1] = glyph[y0 - oy:y1 - oy, :x1 - ox]

    yy = np.sin(np.pi * (np.arange(cell_h) + 0.5) / cell_h) ** 2
    xx = np.sin(np.pi * (np.arange(cell_w) + 0.5) / cell_w) ** 2
    bump = np.outer(yy, xx)
    for i in range(1, len(ALL_CHARS)):
        strength = np.clip((min(i, NR) - 44.0) / 25.0, 0.0, 1.0) * AURA
        soft = _blur(_blur(atlas[i]))
        atlas[i] = np.maximum(atlas[i], soft * SOFTNESS * 1.6) + 255.0 * strength * bump
    return np.clip(atlas, 0.0, 255.0).astype(np.uint32)


class GridRenderer:
    def __init__(self, cols: int, rows: int, cell_w: int, cell_h: int):
        self.cols, self.rows = cols, rows
        self.cell_w, self.cell_h = cell_w, cell_h
        self.width, self.height = cols * cell_w, rows * cell_h
        atlas = build_atlas(cell_w, cell_h)
        self.atlas = atlas.reshape(atlas.shape[0], -1)
        self._mask_rb = np.uint32(0x00FF00FF)
        self._mask_g = np.uint32(0x0000FF00)
        self._shift = np.uint32(8)

        n, run = cols * rows, cell_w * cell_h
        self._masks = np.zeros((n, run), np.uint32)
        self._px = np.zeros((n, run), np.uint32)
        self._tmp = np.zeros((n, run), np.uint32)
        self._pixels = np.zeros((self.height, self.width), np.uint32)
        self._by_cell = self._px.reshape(rows, cols, cell_h, cell_w).transpose(0, 2, 1, 3)
        self._by_line = self._pixels.reshape(rows, cell_h, cols, cell_w)
        self._frame = pygame.image.frombuffer(self._pixels, (self.width, self.height), "RGBX")

    def compose(self, idx: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        c = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint32)
        rb = ((c[:, :, 2] << 16) | c[:, :, 0]).reshape(-1, 1)
        g = (c[:, :, 1] << 8).reshape(-1, 1)
        np.take(self.atlas, idx.reshape(-1), axis=0, out=self._masks, mode="clip")
        np.multiply(self._masks, rb, out=self._px)
        np.right_shift(self._px, self._shift, out=self._px)
        np.bitwise_and(self._px, self._mask_rb, out=self._px)
        np.multiply(self._masks, g, out=self._tmp)
        np.right_shift(self._tmp, self._shift, out=self._tmp)
        np.bitwise_and(self._tmp, self._mask_g, out=self._tmp)
        np.bitwise_or(self._px, self._tmp, out=self._px)
        np.copyto(self._by_line, self._by_cell)
        return self._pixels

    def blit(self, target: pygame.Surface, idx: np.ndarray, rgb: np.ndarray) -> None:
        self.compose(idx, rgb)
        target.blit(self._frame, (0, 0))


COLS, ROWS = 103, 47
CW, CH = 7.0, 15.0

T_FADE_IN = 0.78
T_STRIKE = 1.62
T_OUT = 4.33
T_TIP = 8.90
T_IMPACT = 9.66

WICK_X = 256.0
CANDLE_W = 168.0
CANDLE_TOP = 300.0
CANDLE_BASE = 630.0
CANDLE_H = CANDLE_BASE - CANDLE_TOP
RIM_H = 45.0
FLOOR_Y = 592.0
PIVOT = (WICK_X + CANDLE_W / 2.0, CANDLE_BASE)

FIRE_CX = 59.0
FIRE_PEAK_X = 50.0
RING_CX = 59.5
RING_CY = 30.5

COLON = CHAR_INDEX[":"]
SEMI = CHAR_INDEX[";"]
DOT = CHAR_INDEX["."]
TICK = CHAR_INDEX["'"]
PLUS = CHAR_INDEX["+"]
EQUAL = CHAR_INDEX["="]
STAR = CHAR_INDEX["*"]
AT = CHAR_INDEX["@"]
ZERO = CHAR_INDEX["0"]


def lerp_table(t: float, table_t, table_v) -> float:
    return float(np.interp(t, table_t, table_v))


CAM_T = [0.0, 8.00, 8.18, 8.32, 8.45, 8.59, 8.72, 8.86, 9.00, 9.07, 9.13, 9.20, 9.27, 9.34, 9.41, 9.475, 9.54, 9.61, 9.68, 12.0]
CAM_Z = [1.0, 1.00, 0.97, 0.95, 0.91, 0.89, 0.875, 0.87, 0.865, 0.85, 0.83, 0.803, 0.776, 0.749, 0.722, 0.694, 0.667, 0.648, 0.64, 0.64]
CAM_OX = [0.0, 0.0, -9.0, -17.0, -19.0, -22.0, -25.0, -32.0, -40.0, -40.0, -36.0, -33.0, -30.0, -24.0, -20.0, -20.0, -22.0, -28.0, -40.0, -40.0]
CAM_OY = [0.0, 0.0, 5.0, 10.0, 24.0, 31.0, 33.0, 33.0, 33.0, 37.0, 44.0, 54.0, 62.0, 76.0, 88.0, 97.0, 104.0, 108.0, 108.0, 108.0]

MATCH_T = [1.57, 1.70, 1.77, 1.84, 1.91, 1.98, 2.05, 2.25, 2.39, 2.66, 2.93, 3.07, 3.20, 3.34, 3.48, 3.61, 3.75, 3.89, 4.02, 4.16, 4.29, 4.40]
MATCH_C = [2.0, 3.5, 5.0, 6.5, 8.5, 11.2, 12.3, 12.2, 12.6, 13.0, 18.4, 23.5, 27.5, 31.0, 33.0, 34.0, 34.8, 34.5, 34.0, 27.8, 17.6, 10.0]
MATCH_R = [32.4, 32.0, 31.8, 31.5, 31.2, 31.0, 31.0, 31.0, 31.0, 30.0, 26.0, 23.1, 22.0, 21.0, 21.0, 20.6, 20.0, 20.0, 20.0, 22.5, 24.5, 26.0]

MATCH_H_T = [1.86, 2.00, 2.11, 2.25, 3.40, 3.70, 4.05, 4.16, 4.29, 4.33]
MATCH_H_V = [0.0, 40.0, 72.0, 100.0, 104.0, 118.0, 104.0, 72.0, 28.0, 0.0]
CANDLE_FLAME_T = [4.45, 4.57, 4.84, 5.11, 5.39, 5.66, 5.93, 6.20, 6.75, 7.29, 8.00, 8.90, 9.20, 9.45, 9.66]
CANDLE_FLAME_H = [0.0, 9.0, 24.0, 40.0, 58.0, 76.0, 100.0, 112.0, 134.0, 152.0, 158.0, 152.0, 138.0, 118.0, 84.0]
CANDLE_LIGHT_T = [3.50, 3.78, 4.57, 4.84, 5.11, 5.39, 5.66, 5.93, 6.20, 6.75, 7.30, 9.66]
CANDLE_LIGHT_V = [0.00, 0.30, 0.33, 0.41, 0.50, 0.56, 0.63, 0.72, 0.85, 1.10, 1.45, 1.50]
TONE_GAIN = 0.86
AMBIENT_TONE_T = [1.0, 3.0, 4.6, 6.5]
AMBIENT_TONE_V = [0.22, 0.25, 0.50, 1.00]

TILT_T = [8.90, 9.00, 9.07, 9.13, 9.20, 9.27, 9.34, 9.41, 9.475, 9.54, 9.61, 9.68]
TILT_V = [0.00, 0.07, 0.14, 0.21, 0.32, 0.45, 0.58, 0.80, 1.08, 1.33, 1.51, 1.5708]

WAX_L = np.array([0.00, 0.10, 0.17, 0.28, 0.37, 0.58, 0.77, 1.00, 1.30], dtype=np.float32)
WAX_R = np.array([0.00, 0.07, 0.13, 0.30, 0.47, 0.80, 0.96, 0.92, 0.92], dtype=np.float32)
WAX_G = np.array([0.00, 0.07, 0.12, 0.25, 0.36, 0.51, 0.64, 0.82, 0.90], dtype=np.float32)
WAX_B = np.array([0.00, 0.06, 0.11, 0.21, 0.29, 0.34, 0.39, 0.58, 0.77], dtype=np.float32)


class Sparks:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.cap = capacity
        self.rng = rng
        self.x = np.zeros(capacity, np.float32)
        self.y = np.zeros(capacity, np.float32)
        self.vx = np.zeros(capacity, np.float32)
        self.vy = np.zeros(capacity, np.float32)
        self.age = np.ones(capacity, np.float32)
        self.life = np.ones(capacity, np.float32)
        self.gravity = np.zeros(capacity, np.float32)
        self.drag = np.zeros(capacity, np.float32)
        self.bright = np.ones(capacity, np.float32)
        self.head = 0

    def emit(self, n, x, y, vx, vy, life, gravity=0.0, drag=0.0, bright=1.0):
        n = int(n)
        if n <= 0:
            return
        i = (self.head + np.arange(n)) % self.cap
        self.head = int((self.head + n) % self.cap)
        self.x[i], self.y[i] = x, y
        self.vx[i], self.vy[i] = vx, vy
        self.life[i] = life
        self.age[i] = 0.0
        self.gravity[i] = gravity
        self.drag[i] = drag
        self.bright[i] = bright

    def step(self, dt: float, t: float) -> None:
        alive = self.age < self.life
        if not alive.any():
            return
        self.age[alive] += dt
        self.vy[alive] += self.gravity[alive] * dt
        k = np.exp(-self.drag[alive] * dt)
        self.vx[alive] *= k
        self.vy[alive] *= k
        wobble = (noise2(self.y[alive] * 0.02, t * 1.5 + self.x[alive] * 0.01) - 0.5) * 40.0
        self.x[alive] += (self.vx[alive] + wobble) * dt
        self.y[alive] += self.vy[alive] * dt


class CandleScene:
    def __init__(self, seed: int = 11):
        self.cc, self.rr = np.meshgrid(np.arange(COLS, dtype=np.float32), np.arange(ROWS, dtype=np.float32))
        self.sx = (self.cc + 0.5) * CW
        self.sy = (self.rr + 0.5) * CH
        rng = np.random.default_rng(seed + 1)
        self.n_a = rng.uniform(-1.0, 1.0, (ROWS, COLS)).astype(np.float32)
        self.n_b = rng.random((ROWS, COLS), dtype=np.float32)
        self.n_c = rng.random((ROWS, COLS), dtype=np.float32)
        self.grain = fbm2(self.cc * 0.16 + 5.0, self.rr * 2.7 + 3.0, 3).astype(np.float32)
        self.grain2 = noise2(self.cc * 0.55 + 40.0, self.rr * 3.1 + 11.0).astype(np.float32)
        self.plank = fbm2(self.cc * 0.045 + 90.0, np.floor(self.rr / 3.0) * 1.7 + 7.0, 3).astype(np.float32)

        self.sparks = Sparks(2600, np.random.default_rng(seed + 2))
        self.t = 0.0
        self._strike_acc = 0.0
        self._ember_acc = 0.0
        self._fire_acc = 0.0
        self._burst_done = False
        self.idx = np.zeros((ROWS, COLS), np.int32)
        self.rgb = np.zeros((ROWS, COLS, 3), np.float32)

    def advance(self, t: float) -> None:
        step = 1.0 / 120.0
        if self._burst_done and t - self.t > 1.0:
            self.t = t - 1.0
        while self.t + step <= t:
            self.t += step
            self._simulate(self.t, step)

    def _match_pos(self, t: float):
        c = lerp_table(t, MATCH_T, MATCH_C)
        r = lerp_table(t, MATCH_T, MATCH_R)
        return (c + 0.5) * CW, (r + 0.5) * CH

    def _camera(self, t: float):
        return (
            lerp_table(t, CAM_T, CAM_Z),
            lerp_table(t, CAM_T, CAM_OX),
            lerp_table(t, CAM_T, CAM_OY),
        )

    def _tilt(self, t: float) -> float:
        return lerp_table(t, TILT_T, TILT_V)

    def _candle_top_world(self, t: float):
        th = self._tilt(t)
        px, py = PIVOT
        lx = -CANDLE_W / 2.0
        ly = -CANDLE_H
        return px + np.cos(th) * lx - np.sin(th) * ly, py + np.sin(th) * lx + np.cos(th) * ly

    def _simulate(self, t: float, dt: float) -> None:
        rng = self.sparks.rng

        if T_STRIKE <= t < 2.03:
            self._strike_acc += 700.0 * dt
            n = int(self._strike_acc)
            self._strike_acc -= n
            if n:
                hx, hy = self._match_pos(t)
                a = rng.uniform(0, 2 * np.pi, n)
                sp = rng.uniform(10, 140, n)
                self.sparks.emit(
                    n,
                    hx + rng.normal(0, 12, n),
                    hy - 20 + rng.normal(0, 22, n),
                    np.cos(a) * sp + 70,
                    np.sin(a) * sp + 34,
                    rng.uniform(0.06, 0.28, n),
                    gravity=460.0,
                    drag=2.0,
                )

        if 3.72 <= t < 5.55:
            rate = 34.0 if t < 4.15 else 14.0
            self._ember_acc += rate * dt
            n = int(self._ember_acc)
            self._ember_acc -= n
            if n:
                self.sparks.emit(
                    n,
                    WICK_X + rng.normal(0, 16, n) + 3,
                    CANDLE_TOP - 20 - rng.uniform(0, 45, n),
                    rng.normal(0, 18, n),
                    rng.uniform(-70, -22, n),
                    rng.uniform(0.18, 0.6, n),
                    gravity=-20.0,
                    drag=0.8,
                    bright=rng.uniform(0.35, 1.0, n),
                )

        if t >= T_IMPACT and not self._burst_done:
            self._burst_done = True
            n = 980
            a = rng.uniform(0.0, np.pi * 2.0, n)
            sp = 1180.0 * np.sqrt(rng.random(n))
            sp = np.where(np.sin(a) > 0.0, sp * 0.34, sp)
            cx, cy = (FIRE_CX + 0.5) * CW, 31.0 * CH
            self.sparks.emit(
                n,
                cx + rng.normal(0, 14, n),
                cy + rng.uniform(-6, 26, n),
                np.cos(a) * sp * 0.74,
                np.sin(a) * sp,
                (0.22 + rng.exponential(0.42, n)) * np.where(np.sin(a) > 0.0, 0.55, 1.0),
                gravity=-12.0,
                drag=2.7,
            )

        if t >= T_IMPACT + 0.25:
            self._fire_acc += 44.0 * dt
            n = int(self._fire_acc)
            self._fire_acc -= n
            if n:
                x = rng.normal(FIRE_PEAK_X * CW, 165.0, n)
                self.sparks.emit(
                    n,
                    x,
                    CH * rng.uniform(18.0, 29.0, n),
                    rng.normal(0, 22, n),
                    rng.uniform(-110, -38, n),
                    rng.uniform(0.5, 1.7, n),
                    gravity=-16.0,
                    drag=0.35,
                )

        self.sparks.step(dt, t)

    def render(self, t: float):
        self.advance(t)
        self.idx.fill(0)
        self.rgb.fill(0.0)
        if t < T_FADE_IN:
            return self.idx, self.rgb

        z, ox, oy = self._camera(t)
        wx = (self.sx - ox) / z
        wy = (self.sy - oy) / z
        floor_sy = FLOOR_Y * z + oy

        burning = t >= T_IMPACT
        if burning:
            self._draw_fire_background(t)
        else:
            self._draw_floor_lights(t, wx, floor_sy)
        if t < T_IMPACT + 0.20:
            self._draw_halos(t, wx, wy)
            self._draw_candle(t, wx, wy)
            self._draw_candle_flame(t, wx, wy)
            self._draw_match(t, wx, wy)
        if burning:
            self._draw_fire(t)
        self._draw_sparks(t)
        return self.idx, self.rgb

    def _put(self, mask, idx, r, g, b):
        self.idx[mask] = idx
        self.rgb[mask, 0] = r
        self.rgb[mask, 1] = g
        self.rgb[mask, 2] = b

    def _ramp(self, lum, noise=None, amp=0.0, lo=0, hi=NR):
        v = np.power(np.clip(lum, 0.0, 1.0), 0.7) * NR
        if noise is not None:
            v = v + noise * amp
        return np.clip(v, lo, hi).astype(np.int32)

    def _light_sources(self, t: float):
        lights = []
        gm = self._match_glow(t)
        if gm > 0.01:
            mx, _ = self._match_pos(t)
            lights.append((mx, 0.66 * gm, 126.0))
        h = lerp_table(t, CANDLE_FLAME_T, CANDLE_FLAME_H)
        if h > 1.0:
            fx, _ = self._candle_top_world(t)
            lights.append((fx, 0.92 * (0.25 + min(1.0, h / 120.0)), 150.0))
        return lights

    def _match_glow(self, t: float) -> float:
        if t < T_STRIKE:
            return 0.0
        rise = smoothstep(T_STRIKE, T_STRIKE + 0.2, t)
        return float(rise * (1.0 - smoothstep(4.2, T_OUT, t)))

    def _draw_floor_lights(self, t, wx, floor_sy):
        lights = self._light_sources(t)
        if not lights:
            return
        depth = (self.sy - floor_sy) / CH
        m = depth > -0.5
        total = np.zeros((ROWS, COLS), np.float32)
        for lx, amp, sigma in lights:
            total += amp * np.exp(-(((wx - lx) / sigma) ** 2))
        falloff = np.exp(-np.maximum(depth, 0.0) * 0.46)
        brightness = total * falloff * (0.45 + 1.1 * self.grain) * (0.92 + 0.08 * nz(t * 6.0))
        on = m & (brightness > 0.34) & (depth < 3.2)
        if not on.any():
            return
        is_plus = self.grain2 * 0.45 + self.n_c * 0.55 > 0.50
        v = np.clip(brightness, 0.0, 1.0)
        idx = np.where(is_plus, PLUS, EQUAL)
        v = v * np.exp(-np.maximum(depth, 0.0) * 0.25) * 0.82
        r = np.where(is_plus, 0.62 * v, 0.30 * v)
        g = np.where(is_plus, 0.56 * v, 0.10 * v)
        b = np.where(is_plus, 0.53 * v, 0.10 * v)
        self._put(on, idx[on], r[on], g[on], b[on])

    def _halo(self, wx, wy, fx, fy, rx, ry, strength):
        d = np.sqrt(((wx - fx) / rx) ** 2 + ((wy - fy) / ry) ** 2)
        m = d < 1.0
        if not m.any():
            return
        k = np.clip((1.05 - d) * 1.5, 0.15, 1.0) * strength
        idx = np.where(self.n_b > 0.82, SEMI, COLON)
        v = k * (0.78 + 0.22 * self.n_c)
        self._put(m, idx[m], (0.36 * v)[m], (0.33 * v)[m], (0.28 * v)[m])

    def _draw_halos(self, t, wx, wy):
        h = lerp_table(t, CANDLE_FLAME_T, CANDLE_FLAME_H)
        if h > 70.0:
            fx, fy = self._candle_top_world(t)
            k = float(smoothstep(70.0, 130.0, h))
            if t > T_TIP:
                k = max(k, 0.8)
            self._halo(wx, wy, fx + 2, fy - h * 0.37, 26.0 + 34.0 * k, max(h * 0.40, 44.0), 0.40 + 0.40 * k)
        mh = lerp_table(t, MATCH_H_T, MATCH_H_V)
        if mh > 60.0:
            mx, my = self._match_pos(t)
            self._halo(wx, wy, mx, my - mh * 0.45, 30.0, mh * 0.34, 0.4)

    def _flame(self, t, wx, wy, fx, fy, height, width, lean, g, seed):
        if height < 3.0:
            return
        s = (fy - wy) / height
        region = (s >= 0.0) & (s < 1.0) & (np.abs(wx - fx) < width + height * abs(lean) + 14.0)
        if not region.any():
            return
        sv = np.clip(s[region], 0.0, 1.0)
        flutter = (noise1(sv * 2.3 - t * 3.1 + seed) - 0.5) * 2.0
        cx = fx + lean * height * sv ** 1.4 + flutter * width * 0.24 * sv ** 1.1
        half = width * 0.5 * np.minimum(1.0, 0.72 + 1.6 * sv) * np.power(np.maximum(1.0 - sv ** 2.2, 1e-3), 1.1)
        half = np.maximum(half, 3.6)
        d = np.abs(wx[region] - cx) / half
        inside = d <= 1.0
        if not inside.any():
            return
        idx_cells = np.argwhere(region)[inside]
        rows, cols = idx_cells[:, 0], idx_cells[:, 1]
        sv = sv[inside]
        d = d[inside]
        flick = noise2(cols * 0.7 + 3.0 + seed, rows * 0.7 - t * 7.0)
        lum = g * (0.27 + 0.50 * (1.0 - d * d)) * (1.0 - 0.16 * sv) + (flick - 0.5) * 0.34
        ramp_i = self._ramp(lum, self.n_a[rows, cols], 5.0, lo=19, hi=59)
        r = np.interp(sv, [0.0, 0.3, 0.6, 1.0], [0.64, 0.94, 0.97, 0.86])
        gg = np.interp(sv, [0.0, 0.3, 0.6, 1.0], [0.52, 0.68, 0.58, 0.27])
        b = np.interp(sv, [0.0, 0.3, 0.6, 1.0], [0.42, 0.47, 0.41, 0.24])
        edge = np.clip((d - 0.55) / 0.45, 0.0, 1.0) ** 2.0
        r = r + (0.70 - r) * edge
        gg = gg + (0.17 - gg) * edge
        b = b + (0.15 - b) * edge
        self.idx[rows, cols] = ramp_i
        self.rgb[rows, cols, 0] = r
        self.rgb[rows, cols, 1] = gg
        self.rgb[rows, cols, 2] = b

    def _draw_candle_flame(self, t, wx, wy):
        h = lerp_table(t, CANDLE_FLAME_T, CANDLE_FLAME_H)
        if h <= 0.0 or t >= T_IMPACT + 0.2:
            return
        h = h * (0.90 + 0.2 * nz(t * 2.1, 3.0))
        fx, fy = self._candle_top_world(t)
        wobble = (nz(t * 1.3, 8.0) - 0.5) * 0.26 + 0.05
        width = 24.0 + 0.24 * h
        self._flame(t, wx, wy, fx + 2.0, fy, h, width, wobble, 1.0, 5.0)

    def _draw_match(self, t, wx, wy):
        if t < T_STRIKE - 0.05 or t > T_OUT + 0.04:
            return
        hx, hy = self._match_pos(t)
        glow = self._match_glow(t)
        h = lerp_table(t, MATCH_H_T, MATCH_H_V) * (0.92 + 0.14 * nz(t * 7.0))

        lx, ly = wx - hx, wy - hy
        dirx, diry = -0.64, 0.77
        along = lx * dirx + ly * diry
        across = np.abs(lx * diry - ly * dirx)
        stick = (along > 8.0) & (along < 150.0) & (across < 3.8)
        if stick.any():
            lum = glow * 0.26 * np.exp(-along / 28.0)
            k = lum > 0.03
            m = stick & k
            if m.any():
                i = self._ramp(lum + 0.09, None, lo=3)
                self._put(m, i[m], (lum * 1.1 + 0.05)[m], (lum * 0.85 + 0.04)[m], (lum * 0.6 + 0.03)[m])

        dt = 0.05
        vx = (self._match_pos(t + dt)[0] - self._match_pos(t - dt)[0]) / (2 * dt)
        lean = float(np.clip(-vx / 1400.0, -0.35, 0.35)) + (nz(t * 2.4, 13.0) - 0.5) * 0.18
        self._flame(t, wx, wy, hx + 1.0, hy - 2.0, h, 37.0, lean, 0.98, 21.0)

        m = (np.abs(wx - hx) <= 10.5) & (np.abs(wy - hy) <= 7.5)
        if m.any() and not 3.55 < t < 4.08:
            fade = 0.72 + 0.28 * glow
            ch = np.where(self.n_b > 0.25, AT, ZERO)
            self._put(m, ch[m], 0.95 * fade, 0.25 * fade, 0.17 * fade)

    def _draw_candle(self, t, wx, wy):
        th = self._tilt(t)
        px, py = PIVOT
        dx, dy = wx - px, wy - py
        co, si = np.cos(th), np.sin(th)
        lx = co * dx + si * dy
        ly = -si * dx + co * dy
        u = (lx + CANDLE_W / 2.0) / (CANDLE_W / 2.0)
        v = -ly
        au = np.abs(u)
        inside = (au <= 1.0) & (v >= 0.0)
        top = CANDLE_H - RIM_H * 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - au * au, 0.0)))
        inside &= v <= top
        if not inside.any():
            return

        amb_fade = float(smoothstep(T_FADE_IN, 1.05, t))
        depth = 1.0 - v / CANDLE_H
        is_rim = v > CANDLE_H - RIM_H

        amb = (0.178 - 0.012 * depth) * (1.0 - 0.22 * au * au)
        amb = np.where(is_rim, amb * 1.32, amb)

        gc = lerp_table(t, CANDLE_LIGHT_T, CANDLE_LIGHT_V) * (0.95 + 0.1 * nz(t * 5.3, 2.0))
        d = np.hypot(lx + CANDLE_W / 2.0, (v - (CANDLE_H - 22.0)) * 1.0)
        fall = np.maximum(1.0 - d / 305.0, 0.0)
        normal = np.arcsin(np.clip(u, -1.0, 1.0))
        lambert = np.maximum(np.cos(normal + 0.42), 0.0)
        near = np.maximum(1.0 - 0.86 * au * au - 0.13 * u, 0.0)
        mix = smoothstep(48.0, 112.0, d)
        shade = near + (lambert - near) * mix
        drip = noise1(lx * 0.11 + 40.0)
        streak = 0.66 + 0.68 * smoothstep(0.2, 0.8, drip)
        boost = 1.0 + 0.18 * np.exp(-((d / 80.0) ** 2))
        side = np.minimum((0.80 / 1.32) * gc ** 0.75 * fall * fall * shade * streak * boost, 0.66)
        rim = gc * 1.30 * np.exp(-((d / 64.0) ** 2))
        lit = np.where(is_rim, rim, side)
        lit_tone = np.where(is_rim, gc * 1.30 * np.exp(-((d / 76.0) ** 2)), side)

        base_lit = lit
        gm = self._match_glow(t)
        if gm > 0.01:
            mx, my = self._match_pos(t)
            mh = lerp_table(t, MATCH_H_T, MATCH_H_V)
            dm = np.hypot(wx - mx, (wy - (my - mh * 0.35)))
            ml = gm * np.exp(-((dm / 66.0) ** 2))
            lit = lit + np.where(is_rim, 0.33, 0.14) * ml

        flash = float(smoothstep(T_IMPACT - 0.02, T_IMPACT + 0.02, t))
        if flash > 0.0:
            lit = lit + flash * 0.40

        lum = amb + lit
        ramp_v = np.power(np.clip(lum, 0.0, 1.0), 0.7) * NR + self.n_a * 5.0 + (drip - 0.5) * 9.0
        ramp_i = np.clip(ramp_v, 0, NR).astype(np.int32)
        ramp_i = np.where(lum > 1.12, NR, np.where(lum > 0.98, np.minimum(ramp_i + 5, NR), ramp_i))
        visible = inside & (ramp_v >= 18.5 - np.minimum(lit, 1.0) * 8.0)

        wick = (np.abs(lx + CANDLE_W / 2.0) < 3.7) & is_rim
        visible &= ~wick

        amb_tone = lerp_table(t, AMBIENT_TONE_T, AMBIENT_TONE_V)
        rim_boost = np.where(is_rim, max(amb_tone, 0.47), amb_tone)
        tone = (amb * rim_boost + (lit_tone + lit - base_lit) * TONE_GAIN) * amb_fade
        r = np.interp(tone, WAX_L, WAX_R)
        g = np.interp(tone, WAX_L, WAX_G)
        b = np.interp(tone, WAX_L, WAX_B)
        if flash > 0.0:
            pale = np.clip(1.0 - depth * 1.5, 0.0, 1.0) ** 1.4
            r = r + (0.98 - r) * flash
            g = g + (0.62 + 0.30 * pale - g) * flash
            b = b + (0.36 + 0.34 * pale - b) * flash
        keep = inside & ~visible
        self.idx[keep] = 0
        self.rgb[keep] = 0.0
        self._put(visible, ramp_i[visible], r[visible], g[visible], b[visible])

    def _draw_fire_background(self, t):
        T = t - T_IMPACT
        rr = self.rr
        n_b, n_c = self.n_b, self.n_c

        flash = float(np.interp(T, [0.0, 0.03, 0.14, 0.24, 0.33], [0.0, 1.0, 0.85, 0.30, 0.0]))
        if flash > 0.0:
            rows_on = np.mod(rr, 3.0) < 2.0
            wall = rows_on & (rr < 40) & (self.plank > 0.40)
            k = flash * np.exp(-np.maximum(29.0 - rr, 0.0) / 6.5) * (0.72 + 0.28 * n_c)
            wall &= k > 0.05
            self._put(wall, COLON, (0.80 * k)[wall], (0.77 * k)[wall], (0.72 * k)[wall])

        floor_on = float(smoothstep(0.0, 0.03, T))
        if floor_on > 0.0:
            floor = rr >= 35.0
            third = np.where(np.mod(rr, 3.0) == 2.0, 1.25, 0.95)
            live = floor & (self.grain > 0.34)
            plus = live & (self.grain2 * 0.45 + n_c * 0.55 > 0.595)
            eq = live & ~plus & (n_b > 0.25)
            v = (0.82 + 0.18 * n_c) * third * floor_on * (0.92 + 0.08 * nz(t * 4.0, 9.0))
            v = v * (1.0 + 0.5 * flash)
            self._put(plus, PLUS, (0.38 * v)[plus], (0.27 * v)[plus], (0.23 * v)[plus])
            self._put(eq, EQUAL, (0.10 * v * (1.0 + 0.5 * flash))[eq], (0.045 * v)[eq], (0.04 * v)[eq])

    def _draw_fire(self, t):
        T = t - T_IMPACT
        cc, rr = self.cc, self.rr
        n_a, n_b, n_c = self.n_a, self.n_b, self.n_c

        ring_r = float(np.interp(T, [0.0, 0.12, 0.22, 0.29, 0.36], [0.0, 20.6, 34.0, 49.0, 62.0]))
        front = float(np.interp(T, [0.0, 0.12, 0.40], [0.0, 35.0, 74.0]))
        hole = float(np.interp(T, [0.0, 0.03, 0.06, 0.12, 0.22, 0.29, 0.36, 0.40], [0.0, 0.0, 8.0, 17.0, 27.0, 23.0, 7.0, 0.0]))
        dist = np.abs(cc - FIRE_CX)
        dome = np.sqrt(np.maximum(1.0 - ((rr - 27.5) / max(hole * 0.45, 1.0)) ** 2, 0.0))
        lit_x = (dist < front) & (dist >= hole * dome)
        band_r = rr - float(np.interp(T, [0.0, 0.22, 0.36, 0.60], [2.4, 1.5, 0.7, 0.0]))

        taper = np.clip((front - dist) / 9.0, 0.0, 1.0)

        peak_x = FIRE_PEAK_X + 9.0 * (nz(t * 0.23, 31.0) - 0.5) * 2.0
        rise = float(smoothstep(0.12, 0.42, T))
        surge = 1.0 + 0.35 * float(np.interp(T, [0.3, 0.45, 0.9], [0.0, 1.0, 0.0]))
        mound = (5.0 + 6.5 * nz(t * 0.5, 41.0)) * surge * np.exp(-(((cc - peak_x) / 21.0) ** 2))
        env = 2.2 + 0.8 * rise + mound * rise
        burst = float(np.exp(-T * 2.2)) * float(smoothstep(0.02, 0.09, T))
        env = env + 3.6 * burst * np.exp(-(((dist - front * 0.82) / 7.0) ** 2))
        env = env + 3.2 * burst * np.exp(-(((dist - hole - 3.0) / 5.0) ** 2))
        n1 = fbm2(cc * 0.11 + 11.0, np.full_like(cc, t * 0.62), 3)
        n2 = noise2(cc * 0.42 - t * 0.5, np.full_like(cc, t * 2.3 + 5.0))
        height = env * (0.30 + 1.35 * n1) + 3.2 * (n2 - 0.42)
        height = np.maximum(height, 1.5) * (0.35 + 0.65 * float(smoothstep(0.0, 0.1, T))) * (0.2 + 0.8 * taper)

        above = 29.0 - band_r
        turb = fbm2(cc * 0.33 + np.sin(rr * 0.33 + t) * 0.5, rr * 0.30 + t * 3.1, 2)
        q = above / height + (turb - 0.5) * 1.0
        flame_i = np.clip(1.0 - q, 0.0, 1.0)

        haze_top = (env * 1.32 + 6.0) * (0.25 + 0.75 * rise)
        haze = (above > 0.0) & (above < haze_top + 6.0 * (n_c - 0.5)) & lit_x
        hv = np.clip(1.12 - above / (haze_top + 1.0), 0.10, 1.0) * (0.62 + 0.38 * n_b)
        self._put(haze, np.where(n_b > 0.78, SEMI, COLON)[haze], (0.56 * hv)[haze], (0.51 * hv)[haze], (0.46 * hv)[haze])

        tongue = (above > 0.0) & (flame_i > 0.24) & lit_x
        if tongue.any():
            lum = 0.12 + 0.82 * flame_i ** 1.05
            ri = self._ramp(lum, n_a, 7.0, lo=19, hi=66)
            ci = np.clip(flame_i * 1.3, 0.0, 1.0)
            r = np.interp(ci, [0.0, 0.35, 0.7, 1.0], [0.70, 0.86, 0.97, 1.00])
            g = np.interp(ci, [0.0, 0.35, 0.7, 1.0], [0.27, 0.38, 0.60, 0.78])
            b = np.interp(ci, [0.0, 0.35, 0.7, 1.0], [0.23, 0.27, 0.35, 0.52])
            self._put(tongue, ri[tongue], 0.90 * r[tongue], 0.90 * g[tongue], 0.90 * b[tongue])

        hot = fbm2(cc * 0.17 + 3.0, rr * 0.5 + t * 1.4, 2)
        band_rows = (band_r >= 28.0) & (band_r <= 35.0) & lit_x
        centre = np.exp(-(((cc - peak_x) / 24.0) ** 2))
        base = np.interp(band_r, [28, 29, 30, 31, 32, 33, 34, 35], [52, 59, 68, 70, 70, 50, 29, 11])
        base = base + np.interp(band_r, [27, 28, 29, 30, 31], [0.0, 9.0, 8.0, 2.0, 0.0]) * centre
        spread = np.interp(band_r, [28, 29, 30, 31, 32, 33, 34, 35], [15, 24, 20, 12, 8, 7, 7, 8])
        ri = base + spread * (hot - 0.5) * 2.0 + n_a * np.interp(band_r, [28, 29, 30, 31, 33, 34], [8, 9, 5, 4, 6, 6])
        cream = np.interp(band_r, [28, 29, 30, 31, 32, 33], [0.14, 0.42, 0.88, 1.0, 1.0, 0.0])
        cream = cream + np.interp(band_r, [27, 28, 29, 30], [0.0, 0.42, 0.34, 0.08]) * centre
        cream = np.clip(cream + (hot - 0.5) * np.interp(band_r, [28, 29, 30, 31, 33], [0.9, 1.6, 1.3, 0.6, 0.0]), 0.0, 1.0)
        r0 = np.interp(band_r, [28, 33, 34, 35], [0.98, 0.98, 0.72, 0.42])
        g0 = np.interp(band_r, [28, 33, 34, 35], [0.60, 0.62, 0.33, 0.20])
        b0 = np.interp(band_r, [28, 33, 34, 35], [0.33, 0.36, 0.28, 0.17])
        r = r0 + (0.93 - r0) * cream
        g = g0 + (0.89 - g0) * cream
        b = b0 + (0.76 - b0) * cream
        ri = np.clip(ri, 0, NR).astype(np.int32)
        hot_limit = np.interp(band_r, [28, 29, 30, 31, 32, 33], [69, 68, 67, 65, 65, 69])
        ri = np.where(ri >= hot_limit, AT, ri)
        visible = band_rows & ((band_r < 35.0) | (n_b > 0.42)) & (band_r >= 33.5 - 6.0 * taper)
        self._put(visible, ri[visible], r[visible], g[visible], b[visible])

        fill_rx = min(ring_r * 0.81, hole)
        if fill_rx > 0.5:
            hx = (cc - RING_CX) / fill_rx
            hy = (rr - RING_CY) / (fill_rx * 0.37)
            inner = (hx * hx + hy * hy) < 1.0
            air = inner & (rr + n_a * 1.6 < 32.5)
            hk = 0.20 + 0.14 * n_c
            self._put(air, np.where(n_b > 0.72, SEMI, COLON)[air], (0.75 * hk)[air], (0.62 * hk)[air], (0.5 * hk)[air])
            low = inner & (rr + n_a * 1.6 >= 32.5)
            fade = float(np.interp(T, [0.22, 0.36], [1.0, 0.35]))
            k = (np.interp(rr, [33, 36, 41], [0.50, 0.26, 0.11]) + 0.09 * (hot - 0.5) * 2.0) * fade
            ri2 = self._ramp(k, n_a, 8.0, lo=0, hi=52)
            tint = np.interp(rr, [33, 36, 41], [1.0, 0.62, 0.42])
            show = low & (ri2 >= 7)
            self._put(show, ri2[show], (0.95 * tint)[show], (0.46 * tint)[show], (0.26 * tint)[show])

        ring_a = float(np.interp(T, [0.0, 0.05, 0.22, 0.29, 0.36], [0.0, 1.0, 0.8, 0.34, 0.0]))
        if ring_a > 0.0:
            dx = (cc - RING_CX) * CW
            dy = (rr - RING_CY) * CH
            inner_a = 0.86 * float(np.interp(T, [0.20, 0.27], [1.0, 0.0]))
            for mult, amp in ((1.0, 1.0), (0.81, inner_a)):
                if amp <= 0.02:
                    continue
                rx = ring_r * CW * mult + 1.0
                rho = np.sqrt((dx / rx) ** 2 + (dy / (rx * 0.76)) ** 2)
                band = np.abs(rho - 1.0) * rx < 7.8
                wave = noise2(cc * 0.5, rr * 0.7 + t * 2.0)
                k = ring_a * amp * (0.34 + 0.34 * wave)
                ri3 = self._ramp(k, n_a, 7.0, lo=19, hi=50)
                self._put(band, ri3[band], 0.78 * ring_a + 0.08, 0.31 * ring_a + 0.06, 0.20 * ring_a + 0.05)

    def _draw_sparks(self, t):
        s = self.sparks
        alive = s.age < s.life
        if not alive.any():
            return
        x, y = s.x[alive], s.y[alive]
        col = (x / CW).astype(np.int32)
        row = (y / CH).astype(np.int32)
        ok = (col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS)
        col, row = col[ok], row[ok]
        fade = 1.0 - (s.age[alive] / s.life[alive])[ok]
        tw = 0.72 + 0.28 * noise2(col * 0.9 + t * 9.0, row * 0.9)
        b = np.clip(fade * 1.5, 0.0, 1.0) * tw * s.bright[alive][ok]
        idx = np.where(b > 0.42, STAR, np.where(b > 0.26, PLUS, np.where(b > 0.14, TICK, DOT)))
        self.idx[row, col] = idx
        v = 0.45 + 0.55 * b
        warm = np.clip((row - 26.0) / 8.0, 0.0, 1.0) * (1.0 - float(smoothstep(T_IMPACT + 0.3, T_IMPACT + 0.7, t)))
        warm = warm if t >= T_IMPACT else 0.0
        self.rgb[row, col, 0] = 0.98 * v
        self.rgb[row, col, 1] = (0.88 - 0.34 * warm) * v
        self.rgb[row, col, 2] = (0.75 - 0.42 * warm) * v


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def high_res_timer(enable: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        if enable:
            ctypes.windll.winmm.timeBeginPeriod(1)
        else:
            ctypes.windll.winmm.timeEndPeriod(1)
    except (AttributeError, OSError):
        pass


def pick_cell_size(screen_h: int) -> tuple[int, int]:
    cell_h = max(12, int(screen_h * 0.86) // ROWS)
    cell_w = max(6, round(cell_h * 7.0 / 15.0))
    return cell_w, cell_h


def main() -> int:
    parser = argparse.ArgumentParser(description="ASCII candle animation")
    parser.add_argument("--fps", type=int, default=60, help="target frame rate")
    parser.add_argument("--cell", type=int, default=0, help="character cell height in pixels (0 = automatic)")
    parser.add_argument("--bench", type=float, default=0.0, help="run this many seconds, print the frame rate, exit")
    args = parser.parse_args()

    enable_dpi_awareness()
    pygame.init()
    info = pygame.display.Info()
    cell_w, cell_h = pick_cell_size(info.current_h)
    if args.cell:
        cell_h = args.cell
        cell_w = max(5, round(cell_h * 7.0 / 15.0))

    renderer = GridRenderer(COLS, ROWS, cell_w, cell_h)
    window = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption("Candle")
    pygame.display.set_icon(pygame.Surface((32, 32)))
    window.fill((0, 0, 0))
    pygame.display.flip()

    scene = CandleScene()
    high_res_timer(True)
    frame_time = 1.0 / max(1, args.fps)
    start = time.perf_counter()
    next_frame = start
    running = True
    frames = 0
    worst = 0.0
    last = start
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        scene = CandleScene()
                        start = time.perf_counter()
                        next_frame = start

            now = time.perf_counter()
            if args.bench and now - start >= args.bench:
                print(f"{frames / (now - start):.1f} FPS average, slowest frame {worst * 1000.0:.1f} ms")
                break
            idx, rgb = scene.render(now - start)
            renderer.blit(window, idx, rgb)
            pygame.display.flip()
            frames += 1
            finished = time.perf_counter()
            worst = max(worst, finished - last)
            last = finished

            next_frame += frame_time
            delay = next_frame - time.perf_counter()
            if delay > 0.002:
                time.sleep(delay - 0.001)
            while time.perf_counter() < next_frame:
                pass
            if delay < -0.25:
                next_frame = time.perf_counter()
    finally:
        high_res_timer(False)
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
