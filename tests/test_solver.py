"""
Tests for Solution A, the exact solver in cops_robbers.py.

The solver is also the answer key for scoring Solution B, so each test checks
it against something that does not share its logic:

  1. Hand-worked examples, with answers derived on paper.
  2. A published theorem: one guard can guarantee capture from every intruder
     start (for some guard start) exactly when the graph is dismantlable
     (Quilliot 1978; Nowakowski and Winkler 1983). Checked on every connected
     graph with up to 6 rooms.
  3. The Petersen graph needs exactly 3 guards, so 2 guards always leave an
     escape. This exercises the escape side with 2 guards, as in the dataset.
  4. A different algorithm: depth-limited minimax with iterative deepening
     searches forward from each start, instead of backwards from captures,
     and must give the same capture depth for every room.
  5. Renaming rooms must rename the answers and change nothing else.

Run with: uv run pytest
"""
import itertools
import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cops_robbers import core_tree, ground_truth, solve  # noqa: E402

LOOP = [(0, 1), (1, 2), (2, 3), (3, 0)]  # rooms A-B-C-D-A


def capture_depths(n, edges, guards):
    """Half-moves to capture for each intruder start room, or None if it escapes."""
    truth = ground_truth(n, edges, guards)
    return [truth[r][1] for r in range(n)]


def closed_neighbourhoods(n, edges):
    nbrs = [{v} for v in range(n)]
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return nbrs


def is_connected(n, edges):
    nbrs = closed_neighbourhoods(n, edges)
    seen, stack = {0}, [0]
    while stack:
        for w in nbrs[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def random_connected_graph(n, rng):
    """Random spanning tree plus a few extra corridors."""
    edges = {tuple(sorted((v, rng.randrange(v)))) for v in range(1, n)}
    for _ in range(rng.randint(0, n)):
        a, b = rng.sample(range(n), 2)
        edges.add((min(a, b), max(a, b)))
    return sorted(edges)


# ------------------------------------------------------- 1. hand-worked cases
def test_loop_one_guard():
    # Guard at A. A: caught at the start. B, D: the guard steps in.
    # C: whichever way the guard goes, the intruder steps the other way.
    assert capture_depths(4, LOOP, [0]) == [0, 1, None, 1]


def test_loop_two_guards():
    # Both guards at A. For C they split to B and D (1), the intruder has no
    # safe move (2), and a guard steps onto it (3).
    assert capture_depths(4, LOOP, [0, 0]) == [0, 1, 3, 1]


def test_path_intruder_runs_to_the_dead_end():
    # Path A-B-C-D-E, guard at A. From C, D or E the intruder runs to E and
    # waits; the guard needs 4 moves to get there, with 3 intruder moves in
    # between, so capture takes 7 half-moves.
    path = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert capture_depths(5, path, [0]) == [0, 1, 7, 7, 7]


def test_guards_may_share_a_room():
    # Corridors A-B, A-E, B-C, B-D, C-D, C-E, D-F; both guards start in the
    # side room F. They step into D together (1), then split to B and C (3),
    # which covers every room next to A, B, C and E, so capture takes 5.
    # If the guards could not share a room, one would lag behind and every
    # one of those captures would take 7 half-moves.
    edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)]
    assert capture_depths(6, edges, [5, 5]) == [5, 5, 5, 1, 5, 0]


# -------------------------------------------------- 2. dismantlability theorem
def is_dismantlable(n, edges):
    """Repeatedly delete a room whose neighbourhood is covered by another
    room's; dismantlable if a single room is left."""
    nbrs = closed_neighbourhoods(n, edges)
    alive = set(range(n))
    while len(alive) > 1:
        corner = next((u for u in alive for v in alive
                       if u != v and nbrs[u] & alive <= nbrs[v] & alive), None)
        if corner is None:
            return False
        alive.remove(corner)
    return True


def test_one_guard_matches_dismantlability_on_all_small_graphs():
    checked = 0
    for n in range(1, 7):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            edges = [p for i, p in enumerate(pairs) if mask >> i & 1]
            if not is_connected(n, edges):
                continue
            depth = solve(n, edges, 1)
            guard_wins = any(all(((s,), r, 0) in depth for r in range(n))
                             for s in range(n))
            assert guard_wins == is_dismantlable(n, edges), (n, edges)
            checked += 1
    # Number of connected graphs on 1..6 labelled rooms (OEIS A001187).
    assert checked == 1 + 1 + 4 + 38 + 728 + 26704


# ------------------------------------------------------- 3. Petersen graph
PETERSEN = sorted({tuple(sorted(e)) for i in range(5) for e in
                   [(i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)]})


def test_petersen_two_guards_always_leave_an_escape():
    depth = solve(10, PETERSEN, 2)
    for guards in itertools.combinations_with_replacement(range(10), 2):
        assert any((guards, r, 0) not in depth for r in range(10)), guards


def test_petersen_three_guards_can_always_win_from_some_start():
    depth = solve(10, PETERSEN, 3)
    assert any(all((guards, r, 0) in depth for r in range(10))
               for guards in itertools.combinations_with_replacement(range(10), 3))


# --------------------------------------------- 4. forward minimax cross-check
def minimax_depths(n, edges, guards):
    """Capture depth per intruder start room by iterative-deepening minimax,
    or None if no capture is found within the bound."""
    nbrs = closed_neighbourhoods(n, edges)
    # If the guards can force capture at all, they can do it without the
    # game ever repeating a state, so this bound (more than the number of
    # states) is enough to tell capture from escape.
    bound = n ** len(guards) * n * 2

    @lru_cache(maxsize=None)
    def can_capture(g, i, guards_turn, d):
        """Can the guards force capture within d half-moves?"""
        if i in g:
            return True
        if d == 0:
            return False
        if guards_turn:  # max node: some joint guard move works
            return any(can_capture(tuple(sorted(m)), i, False, d - 1)
                       for m in itertools.product(*(nbrs[x] for x in g)))
        # min node: every intruder move, including staying, still loses
        return all(can_capture(g, j, True, d - 1) for j in nbrs[i])

    start = tuple(sorted(guards))
    return [next((d for d in range(bound + 1) if can_capture(start, r, True, d)), None)
            for r in range(n)]


def test_minimax_agrees_with_solver_on_small_graphs():
    rng = random.Random(4263)
    escapes = 0
    for k, sizes, count in [(1, range(3, 8), 60), (2, range(3, 7), 30)]:
        for _ in range(count):
            n = rng.choice(sizes)
            edges = random_connected_graph(n, rng)
            guards = [rng.randrange(n) for _ in range(k)]
            expected = minimax_depths(n, edges, guards)
            assert capture_depths(n, edges, guards) == expected, (n, edges, guards)
            escapes += expected.count(None)
    assert escapes > 0  # the escape side was exercised too


# ------------------------------------------------------- 5. renaming rooms
def test_renaming_rooms_renames_the_answers():
    rng = random.Random(0)
    for _ in range(5):
        n, edges = core_tree(14, rng)
        guards = [rng.randrange(n), rng.randrange(n)]
        perm = list(range(n))
        rng.shuffle(perm)
        renamed = capture_depths(n, [(perm[a], perm[b]) for a, b in edges],
                                 [perm[g] for g in guards])
        original = capture_depths(n, edges, guards)
        assert all(renamed[perm[r]] == original[r] for r in range(n))
