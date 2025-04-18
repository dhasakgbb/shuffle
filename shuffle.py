import time
import signal
import os
import numpy as np
import numba
import multiprocessing as mp

# Force fork on macOS so child doesn’t immediately exit
mp.set_start_method('fork', force=True)

# --- Configuration & constants ---
UI_REFRESH_HZ    = 60.0           # UI redraw frequency
CHUNK_SIZE       = 50_000         # shuffles per batch
DECK_SIZE        = 52
HIST_BAR_WIDTH   = 35

# 1%‑wide bins 0–100%
SIMILARITY_BINS = list(range(101))
BIN_LABELS      = [f"{i}–{i}%" for i in SIMILARITY_BINS]

# Poker categories EXCLUDING High Card
POKER_CATS = [
    "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Royal Flush"
]

# Precompute display color
from colorsys import hsv_to_rgb
_r, _g, _b = hsv_to_rgb(0.66, 1.0, 1.0)
COLOR_STR = f"rgb({int(_r*255)},{int(_g*255)},{int(_b*255)})"

# LCG constants
_LCG_MULT = np.uint64(6364136223846793005)
_LCG_ADD  = np.uint64(1)
_LCG_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

# bit‐mask for Ace‑high straight (10–J–Q–K–A)
_ROYAL_MASK = (1<<0)|(1<<9)|(1<<10)|(1<<11)|(1<<12)
# bit pattern for five consecutive bits: 0b11111
_CONSEC5    = 0b11111

@numba.njit(nogil=True, fastmath=True, cache=True, boundscheck=False)
def categorize_njit(deck):
    """
    Ultra‑fast 5‑card evaluator:
      returns 0..8 for One Pair ... Royal Flush, -1 for High Card.
    """
    # --- count ranks in a tiny buffer ---
    ur = np.empty(5, np.int32)  # unique ranks
    ct = np.empty(5, np.int32)  # counts
    uniq = 0
    for i in range(5):
        r = deck[i] % 13
        found = False
        for j in range(uniq):
            if ur[j] == r:
                ct[j] += 1
                found = True
                break
        if not found:
            ur[uniq] = r
            ct[uniq] = 1
            uniq += 1

    # --- flush test (unrolled) ---
    s0 = deck[0] // 13
    is_flush = (
        deck[1]//13 == s0 and deck[2]//13 == s0 and
        deck[3]//13 == s0 and deck[4]//13 == s0
    )

    # --- straight test via bit‑mask ---
    mask = 0
    for i in range(5):
        mask |= 1 << (deck[i] % 13)
    is_straight = False
    if uniq == 5:
        # check any run of five
        for shift in range(9):
            if ((mask >> shift) & _CONSEC5) == _CONSEC5:
                is_straight = True
                break
        # wheel or royal: also treat 10-J-Q-K-A as straight
        if not is_straight and (mask & _ROYAL_MASK) == _ROYAL_MASK:
            is_straight = True

    # --- count duplicates ---
    maxc = 0
    pairs = 0
    for j in range(uniq):
        cj = ct[j]
        if cj > maxc:
            maxc = cj
        if cj == 2:
            pairs += 1

    # --- decide category ---
    if is_straight and is_flush:
        return 8 if (mask & _ROYAL_MASK) == _ROYAL_MASK else 7
    if maxc == 4:
        return 6
    if maxc == 3 and pairs == 1:
        return 5
    if is_flush:
        return 4
    if is_straight:
        return 3
    if maxc == 3:
        return 2
    if pairs == 2:
        return 1
    if pairs == 1:
        return 0
    return -1


@numba.njit(nogil=True, fastmath=True, cache=True, boundscheck=False)
def simulate_and_poker(
    deck,
    hist, total_arr, rises_arr, poker_arr,
    n_steps, rng_state
):
    d    = deck
    h    = hist
    t    = total_arr
    r_out= rises_arr
    p    = poker_arr
    st   = rng_state
    N    = d.shape[0]
    B    = h.shape[0]

    for _ in range(n_steps):
        # --- Fisher–Yates via inline LCG ---
        for i in range(N-1, 0, -1):
            x = (st[0] * _LCG_MULT + _LCG_ADD) & _LCG_MASK
            st[0] = x
            j = x % (i + 1)
            tmp = d[i]; d[i] = d[j]; d[j] = tmp

        # positional similarity
        t[0] += 1
        m = 0
        for k in range(N):
            if d[k] == k:
                m += 1

        # rising sequences
        r = 1
        prev = d[0]
        for k in range(1, N):
            cur = d[k]
            if cur < prev:
                r += 1
            prev = cur
        r_out[0] = r

        # histogram bin
        idx = (m * 100 // N)
        if idx >= B:
            idx = B - 1
        h[idx] += 1

        # poker on top‑5 cards
        cat = categorize_njit(d)
        if cat >= 0:
            p[cat] += 1


def shuffle_worker(
    hist_buf, total_buf, rises_buf, poker_buf, rng_buf, stop_evt
):
    hist      = np.frombuffer(hist_buf, dtype=np.int32)
    total     = np.frombuffer(total_buf, dtype=np.int32)
    rises     = np.frombuffer(rises_buf, dtype=np.int32)
    poker     = np.frombuffer(poker_buf, dtype=np.int64)
    rng_state = np.frombuffer(rng_buf, dtype=np.uint64)

    deck = np.arange(DECK_SIZE, dtype=np.int8)

    # warm up
    simulate_and_poker(
        deck.copy(), hist.copy(), total.copy(),
        rises.copy(), poker.copy(), 10, rng_state.copy()
    )

    while not stop_evt.is_set():
        simulate_and_poker(
            deck, hist, total, rises,
            poker, CHUNK_SIZE, rng_state
        )


def render_hist(sim):
    from rich.panel import Panel
    maxc = sim.hist.max() or 1
    lines = []
    for i, cnt in enumerate(sim.hist):
        if cnt == 0:
            continue
        width = max(1, cnt * HIST_BAR_WIDTH // maxc)
        bar   = "█" * width
        lines.append(f"{BIN_LABELS[i]:>6s} │ [{COLOR_STR}]{bar:<{HIST_BAR_WIDTH}}[/] ({cnt:,})")
    return Panel("\n".join(lines), title="Positional Similarity", border_style="cyan")


def render_stats(sim, elapsed):
    from rich.panel import Panel
    from rich.table import Table
    rate     = sim.total[0] / elapsed if elapsed > 0 else 0.0
    mids     = np.array(SIMILARITY_BINS)
    mean_sim = (sim.hist.astype(np.int64) * mids).sum() / sim.total[0] if sim.total[0] else 0.0
    cum      = np.cumsum(sim.hist)
    half     = sim.total[0] // 2
    mb_idx   = int(np.searchsorted(cum, half))
    mb       = SIMILARITY_BINS[min(mb_idx, len(SIMILARITY_BINS)-1)]

    tbl = Table(show_header=False, box=None, padding=(0,1))
    tbl.add_column(justify="right", style="bold cyan")
    tbl.add_column(justify="left",  style="bold magenta")
    tbl.add_row("Total:",      f"{sim.total[0]:,}")
    tbl.add_row("Rate:",       f"{rate:,.1f} shuffles/sec")
    tbl.add_row("Mean Sim.:",  f"{mean_sim:.2f}%")
    tbl.add_row("Median Bin:", f"{mb}–{mb}%")
    tbl.add_row("Rising Seqs:",f"{sim.rises[0]}")
    return Panel(tbl, title="Statistics", border_style="green")


def render_poker(sim):
    from rich.panel import Panel
    from rich.table import Table
    total = sim.total[0] or 1
    tbl   = Table(show_header=True, box=None, padding=(0,1))
    tbl.add_column("Hand",  style="bold yellow")
    tbl.add_column("Count", justify="right")
    tbl.add_column("%",     justify="right")
    for name, cnt in zip(POKER_CATS, sim.poker):
        pct = cnt * 100 / total
        tbl.add_row(name, f"{cnt:,}", f"{pct:.2f}%")
    return Panel(tbl, title="5‑Card Hands", border_style="yellow")


class DashboardState:
    def __init__(self, hist_buf, total_buf, rises_buf, poker_buf):
        self.hist  = np.frombuffer(hist_buf, dtype=np.int32)
        self.total = np.frombuffer(total_buf, dtype=np.int32)
        self.rises = np.frombuffer(rises_buf, dtype=np.int32)
        self.poker = np.frombuffer(poker_buf, dtype=np.int64)


def main():
    print("🚀 Starting shuffle+5‑card poker dashboard. Ctrl‑C to quit.", flush=True)

    hist_buf  = mp.RawArray("i", len(SIMILARITY_BINS))
    total_buf = mp.RawArray("i", 1)
    rises_buf = mp.RawArray("i", 1)
    poker_buf = mp.RawArray("l", len(POKER_CATS))
    rng_buf   = mp.RawArray("Q", 1)

    rng = np.frombuffer(rng_buf, dtype=np.uint64)
    rng[0] = np.uint64(int.from_bytes(os.urandom(8), "little"))

    stop_evt = mp.Event()

    worker = mp.Process(
        target=shuffle_worker,
        args=(hist_buf, total_buf, rises_buf, poker_buf, rng_buf, stop_evt),
        daemon=True
    )
    worker.start()

    sim = DashboardState(hist_buf, total_buf, rises_buf, poker_buf)
    signal.signal(signal.SIGINT, lambda s,f: stop_evt.set())

    from rich.live import Live
    from rich.layout import Layout
    from rich.table import Table
    from time import perf_counter

    layout = Layout()
    layout.split_row(
        Layout(name="hist", ratio=2),
        Layout(name="side", ratio=1),
    )
    layout["side"].split_column(
        Layout(name="stats", ratio=1),
        Layout(name="poker", ratio=2),
    )

    start = perf_counter()
    with Live(layout, refresh_per_second=UI_REFRESH_HZ, screen=True):
        while not stop_evt.is_set():
            elapsed = perf_counter() - start
            layout["hist"].update(render_hist(sim))
            layout["stats"].update(render_stats(sim, elapsed))
            layout["poker"].update(render_poker(sim))

    worker.join(timeout=1.0)
    print("\nBye!")


if __name__ == "__main__":
    main()
