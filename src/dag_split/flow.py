from collections import deque
from dataclasses import dataclass
from typing import List, Tuple


EPS = 1e-10


@dataclass
class _Edge:
    to: int
    rev: int
    cap: float


class Dinic:
    """Directed max-flow/min-cut with floating-point capacities."""

    def __init__(self, n: int) -> None:
        self.graph: List[List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, src: int, dst: int, cap: float) -> None:
        if cap < -EPS:
            raise ValueError(f"capacity must be non-negative, got {cap}")
        fwd = _Edge(dst, len(self.graph[dst]), float(max(cap, 0.0)))
        rev = _Edge(src, len(self.graph[src]), 0.0)
        self.graph[src].append(fwd)
        self.graph[dst].append(rev)

    def _bfs(self, source: int, sink: int) -> List[int]:
        level = [-1] * len(self.graph)
        level[source] = 0
        q = deque([source])
        while q:
            v = q.popleft()
            for edge in self.graph[v]:
                if edge.cap > EPS and level[edge.to] < 0:
                    level[edge.to] = level[v] + 1
                    q.append(edge.to)
        return level

    def _dfs(
        self,
        v: int,
        sink: int,
        pushed: float,
        level: List[int],
        it: List[int],
    ) -> float:
        if v == sink:
            return pushed
        while it[v] < len(self.graph[v]):
            edge = self.graph[v][it[v]]
            if edge.cap > EPS and level[v] + 1 == level[edge.to]:
                flow = self._dfs(edge.to, sink, min(pushed, edge.cap), level, it)
                if flow > EPS:
                    edge.cap -= flow
                    self.graph[edge.to][edge.rev].cap += flow
                    return flow
            it[v] += 1
        return 0.0

    def max_flow(self, source: int, sink: int) -> float:
        total = 0.0
        while True:
            level = self._bfs(source, sink)
            if level[sink] < 0:
                break
            it = [0] * len(self.graph)
            while True:
                pushed = self._dfs(source, sink, float("inf"), level, it)
                if pushed <= EPS:
                    break
                total += pushed
        return total

    def reachable_from(self, source: int) -> List[bool]:
        seen = [False] * len(self.graph)
        seen[source] = True
        q = deque([source])
        while q:
            v = q.popleft()
            for edge in self.graph[v]:
                if edge.cap > EPS and not seen[edge.to]:
                    seen[edge.to] = True
                    q.append(edge.to)
        return seen


def min_cut(n: int, edges: List[Tuple[int, int, float]], source: int, sink: int) -> Tuple[float, List[bool]]:
    solver = Dinic(n)
    for src, dst, cap in edges:
        solver.add_edge(src, dst, cap)
    value = solver.max_flow(source, sink)
    return value, solver.reachable_from(source)
