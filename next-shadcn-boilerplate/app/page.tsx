"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type Agent = { id: number | string; embedding: number[] };
type Edge = { a: number | string; b: number | string; w: number };
type Last = { a: number | string; b: number | string; action: string } | null;
type Metrics = {
  cohesion: number;
  polarization: number;
  spread_pc1: number;
  mean_edge_weight: number;
  trait_means: Record<string, number>;
  trait_stds: Record<string, number>;
  action_counts: Record<string, number>;
};
type Traits = { optimism: number; sociability: number; risk: number; consumerism: number };
type Planner = "homophily" | "round_robin" | "nearest" | "tension";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const WORLD_W = 2000;
const WORLD_H = 1400;
const GRID = 64;

const embToTraits = (e: number[] = []): Traits => ({
  optimism: e[0] ?? 0,
  sociability: e[1] ?? 0,
  risk: e[2] ?? 0,
  consumerism: e[3] ?? 0,
});
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export default function SyntheticSocietyPage() {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [agents, setAgents] = useState<Agent[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [last, setLast] = useState<Last>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);

  const [selectedId, setSelectedId] = useState<number | string | null>(null);
  const selectedAgent = agents.find((a) => String(a.id) === String(selectedId)) || null;
  const [traits, setTraits] = useState<Traits>(embToTraits());

  // overlays
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [showRegions, setShowRegions] = useState(true);

  // visibility tuning
  const [heatmapIntensity, setHeatmapIntensity] = useState(1.2); // 0.5–2.0
  const [edgeOpacity, setEdgeOpacity] = useState(1.0); // 0.4–1.6
  const [contrast, setContrast] = useState(1.0); // 0.8–1.5
  const [focusSelected, setFocusSelected] = useState(false);

  // fullscreen
  const [isFullscreen, setIsFullscreen] = useState(false);

  // planner ui
  const [planner, setPlanner] = useState<Planner>("homophily");

  // camera
  const cameraRef = useRef({ x: 0, y: 0, scale: 1 });
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 6;

  // draw refs
  const stateRef = useRef({
    agents,
    edges,
    last,
    selectedId,
    metrics,
    showHeatmap,
    showAxes,
    showRegions,
    heatmapIntensity,
    edgeOpacity,
    contrast,
    focusSelected,
  });
  useEffect(() => {
    stateRef.current = {
      agents,
      edges,
      last,
      selectedId,
      metrics,
      showHeatmap,
      showAxes,
      showRegions,
      heatmapIntensity,
      edgeOpacity,
      contrast,
      focusSelected,
    };
  }, [
    agents,
    edges,
    last,
    selectedId,
    metrics,
    showHeatmap,
    showAxes,
    showRegions,
    heatmapIntensity,
    edgeOpacity,
    contrast,
    focusSelected,
  ]);

  const posRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  // API pickers
  const pickAgents = (d: any): Agent[] => d?.agents ?? d?.state?.agents ?? [];
  const pickEdges = (d: any): Edge[] => d?.edges ?? d?.state?.edges ?? [];
  const pickLast = (d: any): Last => d?.last ?? d?.state?.last ?? null;
  const pickMetrics = (d: any): Metrics | null => d?.metrics ?? d?.state?.metrics ?? null;
  const pickEvents = (d: any): string[] => d?.events ?? d?.state?.events ?? [];

  const fetchState = async () => {
    const res = await fetch(`${API_BASE}/state`);
    if (!res.ok) return;
    const data = await res.json();
    const as = pickAgents(data);
    setAgents(as);
    setEdges(pickEdges(data));
    setLast(pickLast(data));
    setMetrics(pickMetrics(data));
    setEvents(pickEvents(data));
    if (selectedId != null) {
      const a = as.find((x) => String(x.id) === String(selectedId));
      if (a) setTraits(embToTraits(a.embedding));
    }
    fitCameraToWorld();
  };

  const step = async () => {
    const res = await fetch(`${API_BASE}/step`, { method: "POST" });
    if (!res.ok) return;
    const data = await res.json();
    const as = pickAgents(data);
    setAgents(as);
    setEdges(pickEdges(data));
    setLast(pickLast(data));
    setMetrics(pickMetrics(data));
    setEvents(pickEvents(data));
    setLogs((prev) => [JSON.stringify({ t: Date.now(), last: data.last }), ...prev].slice(0, 120));
    if (selectedId != null) {
      const a = as.find((x) => String(x.id) === String(selectedId));
      if (a) setTraits(embToTraits(a.embedding));
    }
  };

  const reset = async () => {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
    setSelectedId(null);
    await fetchState();
  };

  const applyTraits = async () => {
    if (selectedId == null) return;
    const res = await fetch(`${API_BASE}/agents/${selectedId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ traits }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const as = pickAgents(data);
    setAgents(as);
    setEdges(pickEdges(data));
    setLast(pickLast(data));
    setMetrics(pickMetrics(data));
    setEvents(pickEvents(data));
  };

  const sendEvent = async (type: string) => {
    const res = await fetch(`${API_BASE}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });
    if (!res.ok) return;
    const data = await res.json();
    setAgents(pickAgents(data));
    setEdges(pickEdges(data));
    setLast(pickLast(data));
    setMetrics(pickMetrics(data));
    setEvents(pickEvents(data));
  };

  // switch planner
  const applyPlanner = async (p: Planner) => {
    setPlanner(p);
    await fetch(`${API_BASE}/planner`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy: p }),
    });
  };

  useEffect(() => {
    fetchState();
  }, []);

  // autorun
  useEffect(() => {
    if (!running) return;
    const iv = setInterval(step, Math.max(33, 1000 / speed));
    return () => clearInterval(iv);
  }, [running, speed]);

  // fullscreen
  useEffect(() => {
    const onFSChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFSChange);
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "f") toggleFullscreen();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("fullscreenchange", onFSChange);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  const toggleFullscreen = () => {
    const el = wrapRef.current;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen?.({ navigationUI: "hide" } as any);
    else document.exitFullscreen?.();
  };

  // map embeddings -> world coords
  const targets = useMemo(() => {
    const xs = agents.map((a) => a.embedding?.[0] ?? 0);
    const ys = agents.map((a) => a.embedding?.[1] ?? 0);
    const minx = Math.min(...xs, -1),
      maxx = Math.max(...xs, 1);
    const miny = Math.min(...ys, -1),
      maxy = Math.max(...ys, 1);
    const pad = 80;
    const mapX = (x: number) => pad + ((x - minx) / (maxx - minx + 1e-6)) * (WORLD_W - 2 * pad);
    const mapY = (y: number) => pad + ((y - miny) / (maxy - miny + 1e-6)) * (WORLD_H - 2 * pad);
    return { mapX, mapY, minx, maxx, miny, maxy };
  }, [agents]);

  // canvas + interactions
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const fitCanvas = () => {
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      const rect = wrap.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    const ro = new ResizeObserver(fitCanvas);
    ro.observe(wrap);
    fitCanvas();

    // PAN & ZOOM
    let dragging = false;
    let lastMouseX = 0,
      lastMouseY = 0;

    const worldFromScreen = (sx: number, sy: number) => {
      const { x, y, scale } = cameraRef.current;
      return { wx: x + sx / scale, wy: y + sy / scale };
    };

    const zoomAt = (factor: number, sx: number, sy: number) => {
      const cam = cameraRef.current;
      const before = worldFromScreen(sx, sy);
      cam.scale = clamp(cam.scale * factor, MIN_SCALE, MAX_SCALE);
      const after = worldFromScreen(sx, sy);
      cam.x += before.wx - after.wx;
      cam.y += before.wy - after.wy;
      clampCamera();
    };

    const clampCamera = () => {
      const cam = cameraRef.current;
      const vw = (canvas.clientWidth || 1) / cam.scale;
      const vh = (canvas.clientHeight || 1) / cam.scale;
      cam.x = clamp(cam.x, 0, Math.max(0, WORLD_W - vw));
      cam.y = clamp(cam.y, 0, Math.max(0, WORLD_H - vh));
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey) {
        const factor = Math.exp(-e.deltaY * 0.002);
        const rect = canvas.getBoundingClientRect();
        zoomAt(factor, e.clientX - rect.left, e.clientY - rect.top);
      } else {
        const cam = cameraRef.current;
        cam.x += e.deltaX / cam.scale;
        cam.y += e.deltaY / cam.scale;
        clampCamera();
      }
    };

    const onMouseDown = (e: MouseEvent) => {
      dragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging) return;
      const cam = cameraRef.current;
      cam.x -= (e.clientX - lastMouseX) / cam.scale;
      cam.y -= (e.clientY - lastMouseY) / cam.scale;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      clampCamera();
    };
    const onMouseUp = () => {
      dragging = false;
    };

    // touch (1-finger pan, 2-finger pinch)
    let touchMode: "none" | "pan" | "pinch" = "none";
    let lastTouchX = 0,
      lastTouchY = 0,
      lastDist = 0;
    const distance = (t1: Touch, t2: Touch) => Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
    const midpoint = (t1: Touch, t2: Touch) => ({ x: (t1.clientX + t2.clientX) / 2, y: (t1.clientY + t2.clientY) / 2 });

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        touchMode = "pan";
        lastTouchX = e.touches[0].clientX;
        lastTouchY = e.touches[0].clientY;
      } else if (e.touches.length >= 2) {
        touchMode = "pinch";
        lastDist = distance(e.touches[0], e.touches[1]);
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      if (touchMode === "pan" && e.touches.length === 1) {
        const t = e.touches[0];
        const cam = cameraRef.current;
        cam.x -= (t.clientX - lastTouchX) / cam.scale;
        cam.y -= (t.clientY - lastTouchY) / cam.scale;
        lastTouchX = t.clientX;
        lastTouchY = t.clientY;
        clampCamera();
      } else if (touchMode === "pinch" && e.touches.length >= 2) {
        const d = distance(e.touches[0], e.touches[1]);
        const f = d / (lastDist || d);
        const rect = canvas.getBoundingClientRect();
        const m = midpoint(e.touches[0], e.touches[1]);
        zoomAt(f, m.x - rect.left, m.y - rect.top);
        lastDist = d;
      }
    };
    const onTouchEnd = () => {
      touchMode = "none";
    };

    // click select agent
    const onClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const { wx, wy } = worldFromScreen(sx, sy);
      let found: Agent | null = null;
      stateRef.current.agents.forEach((a) => {
        const p = posRef.current.get(String(a.id));
        if (!p) return;
        const r = Math.max(6, 12 + Math.tanh(a.embedding?.[0] ?? 0) * 10) + 6; // size by optimism
        const dx = wx - p.x,
          dy = wy - p.y;
        if (dx * dx + dy * dy <= r * r) found = a;
      });
      if (found) {
        setSelectedId(found.id);
        setTraits(embToTraits(found.embedding));
      }
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);
    canvas.addEventListener("click", onClick);

    // RAF
    let raf = 0;
    const tick = () => {
      draw(ctx, canvas);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("wheel", onWheel as any);
      canvas.removeEventListener("mousedown", onMouseDown as any);
      window.removeEventListener("mousemove", onMouseMove as any);
      window.removeEventListener("mouseup", onMouseUp as any);
      canvas.removeEventListener("touchstart", onTouchStart as any);
      canvas.removeEventListener("touchmove", onTouchMove as any);
      canvas.removeEventListener("touchend", onTouchEnd as any);
      canvas.removeEventListener("click", onClick as any);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fitCameraToWorld = () => {
    const cam = cameraRef.current;
    const cw = canvasRef.current?.clientWidth || 1;
    const ch = canvasRef.current?.clientHeight || 1;
    const sx = cw / WORLD_W;
    const sy = ch / WORLD_H;
    cam.scale = clamp(Math.min(sx, sy), MIN_SCALE, MAX_SCALE);
    cam.x = 0;
    cam.y = 0;
  };

  // smooth toward targets (less smoothing so motion reads)
  useEffect(() => {
    const cur = posRef.current;
    agents.forEach((a) => {
      const id = String(a.id);
      const prev = cur.get(id) || { x: Math.random() * WORLD_W, y: Math.random() * WORLD_H };
      const tx = targets.mapX(a.embedding?.[0] ?? 0);
      const ty = targets.mapY(a.embedding?.[1] ?? 0);
      cur.set(id, { x: lerp(prev.x, tx, 0.25), y: lerp(prev.y, ty, 0.25) }); // was 0.12
    });
  }, [agents, targets]);

  // drawing
  const draw = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const w = canvas.clientWidth,
      h = canvas.clientHeight;
    const {
      agents,
      edges,
      last,
      selectedId,
      metrics,
      showHeatmap,
      showAxes,
      showRegions,
      heatmapIntensity,
      edgeOpacity,
      contrast,
      focusSelected,
    } = stateRef.current;
    const cam = cameraRef.current;
    const px = (v: number) => v / (cam.scale * dpr); // desired screen px -> world units

    // clear + subtle gradient
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w * dpr, h * dpr);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w * dpr, h * dpr);
    const grad = ctx.createLinearGradient(0, 0, 0, h * dpr);
    grad.addColorStop(0, "rgba(0,0,0,0.03)");
    grad.addColorStop(1, "rgba(0,0,0,0.00)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w * dpr, h * dpr);

    // world transform
    ctx.setTransform(cam.scale * dpr, 0, 0, cam.scale * dpr, -cam.x * cam.scale * dpr, -cam.y * cam.scale * dpr);

    const viewW = w / cam.scale,
      viewH = h / cam.scale;
    const startX = Math.floor(cam.x / GRID) * GRID,
      endX = cam.x + viewW + GRID;
    const startY = Math.floor(cam.y / GRID) * GRID,
      endY = cam.y + viewH + GRID;

    // grid
    ctx.save();
    ctx.strokeStyle = `rgba(0,0,0,${0.08 * contrast})`;
    ctx.lineWidth = px(1);
    for (let gx = startX; gx <= endX; gx += GRID) {
      ctx.beginPath();
      ctx.moveTo(gx, cam.y - GRID);
      ctx.lineTo(gx, cam.y + viewH + GRID);
      ctx.stroke();
    }
    for (let gy = startY; gy <= endY; gy += GRID) {
      ctx.beginPath();
      ctx.moveTo(cam.x - GRID, gy);
      ctx.lineTo(cam.x + viewH + GRID, gy);
      ctx.stroke();
    }
    ctx.restore();

    // heatmap
    if (showHeatmap) {
      const cell = 96;
      const counts = new Map<string, number>();
      let maxC = 0;
      agents.forEach((a) => {
        const p = posRef.current.get(String(a.id));
        if (!p) return;
        const ix = Math.floor(p.x / cell),
          iy = Math.floor(p.y / cell);
        const key = ix + "," + iy;
        const c = (counts.get(key) || 0) + 1;
        counts.set(key, c);
        if (c > maxC) maxC = c;
      });
      counts.forEach((c, key) => {
        const [ix, iy] = key.split(",").map(Number);
        const x = ix * cell,
          y = iy * cell;
        if (x < cam.x - cell || x > cam.x + viewW + cell || y < cam.y - cell || y > cam.y + viewH + cell) return;
        const t = c / Math.max(1, maxC);
        const pow = Math.pow(t, 0.75);
        const hue = 220 - 220 * pow; // blue -> red
        const alpha = (0.09 + 0.28 * pow) * heatmapIntensity;
        ctx.fillStyle = `hsla(${hue},90%,50%,${alpha})`;
        ctx.fillRect(x, y, cell, cell);
      });
    }

    // region tints
    if (showRegions) {
      const zeroX = targets.mapX(0);
      const zeroY = targets.mapY(0);
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.03)";
      ctx.fillRect(zeroX, cam.y - GRID, cam.x + viewW + GRID - zeroX, zeroY - (cam.y - GRID)); // Q1
      ctx.fillRect(zeroX, zeroY, cam.x + viewW + GRID - zeroX, cam.y + viewH + GRID - zeroY); // Q2
      ctx.fillRect(cam.x - GRID, cam.y - GRID, zeroX - (cam.x - GRID), zeroY - (cam.y - GRID)); // Q3
      ctx.fillRect(cam.x - GRID, zeroY, zeroX - (cam.x - GRID), cam.y + viewH + GRID - zeroY); // Q4
      ctx.restore();
    }

    // edges
    edges.forEach((e) => {
      const pa = posRef.current.get(String(e.a));
      const pb = posRef.current.get(String(e.b));
      if (!pa || !pb) return;
      const strength = Math.min(1, 0.15 + Math.log1p(e.w) * 0.25);
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0,0,0,${(0.12 + strength * 0.25) * edgeOpacity})`;
      ctx.lineWidth = px(1 + strength * 3);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    });

    // last interaction
    if (last) {
      const pa = posRef.current.get(String(last.a));
      const pb = posRef.current.get(String(last.b));
      if (pa && pb) {
        ctx.save();
        ctx.setLineDash([px(8), px(8)]);
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.lineWidth = px(3);
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
        ctx.restore();
      }
    }

    // axes + ticks
    if (showAxes || showRegions) {
      const zeroX = targets.mapX(0);
      const zeroY = targets.mapY(0);
      if (showAxes) {
        ctx.save();
        ctx.strokeStyle = `rgba(0,0,0,${0.45 * contrast})`;
        ctx.lineWidth = px(2);

        ctx.beginPath();
        ctx.moveTo(cam.x - GRID, zeroY);
        ctx.lineTo(cam.x + viewW + GRID, zeroY);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(zeroX, cam.y - GRID);
        ctx.lineTo(zeroX, cam.y + viewH + GRID);
        ctx.stroke();

        // ticks -3..3
        ctx.fillStyle = `rgba(0,0,0,${0.75 * contrast})`;
        const labelPx = Math.max(10, 12);
        ctx.font = `${labelPx / cam.scale}px ui-sans-serif`;
        for (let v = -3; v <= 3; v++) {
          const x = targets.mapX(v);
          ctx.beginPath();
          ctx.moveTo(x, zeroY - px(6));
          ctx.lineTo(x, zeroY + px(6));
          ctx.stroke();
          if (Math.abs(v) === 3 || v === 0) ctx.fillText(String(v), x + px(4), zeroY - px(8));

          const y = targets.mapY(v);
          ctx.beginPath();
          ctx.moveTo(zeroX - px(6), y);
          ctx.lineTo(zeroX + px(6), y);
          ctx.stroke();
        }

        // arrows
        const ah = px(12);
        // →
        ctx.beginPath();
        ctx.moveTo(cam.x + viewW + GRID, zeroY);
        ctx.lineTo(cam.x + viewW + GRID - ah, zeroY - ah / 2);
        ctx.lineTo(cam.x + viewW + GRID - ah, zeroY + ah / 2);
        ctx.closePath();
        ctx.fill();
        // ↑
        ctx.beginPath();
        ctx.moveTo(zeroX, cam.y - GRID);
        ctx.lineTo(zeroX - ah / 2, cam.y - GRID + ah);
        ctx.lineTo(zeroX + ah / 2, cam.y - GRID + ah);
        ctx.closePath();
        ctx.fill();

        ctx.fillText("optimism →", cam.x + viewW - px(120), zeroY - px(10));
        ctx.save();
        ctx.translate(zeroX + px(10), cam.y + px(20));
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("sociability ↑", 0, 0);
        ctx.restore();

        ctx.restore();
      }

      if (showRegions) {
        ctx.save();
        ctx.fillStyle = `rgba(0,0,0,${0.55 * contrast})`;
        const pad = 80;
        const q1 = { x: zeroX + pad, y: zeroY - pad, label: "Optimistic • Social" };
        const q2 = { x: zeroX + pad, y: zeroY + pad, label: "Optimistic • Solitary" };
        const q3 = { x: zeroX - pad, y: zeroY - pad, label: "Pessimistic • Social" };
        const q4 = { x: zeroX - pad, y: zeroY + pad, label: "Pessimistic • Solitary" };
        const labelPx = Math.max(11, 13);
        ctx.font = `${labelPx / cam.scale}px ui-sans-serif`;
        [q1, q2, q3, q4].forEach((q) => {
          const tw = ctx.measureText(q.label).width;
          ctx.fillStyle = "rgba(255,255,255,0.85)";
          ctx.fillRect(q.x - tw / 2 - px(6), q.y - px(14), tw + px(12), px(20));
          ctx.fillStyle = `rgba(0,0,0,${0.8 * contrast})`;
          ctx.fillText(q.label, q.x - tw / 2, q.y);
        });
        ctx.restore();
      }
    }

    // nodes
    agents.forEach((a) => {
      const p = posRef.current.get(String(a.id));
      if (!p) return;
      const e0 = a.embedding?.[0] ?? 0,
        e1 = a.embedding?.[1] ?? 0,
        e2 = a.embedding?.[2] ?? 0;
      const r = Math.max(6, 12 + Math.tanh(e0) * 10);
      const hue = (Math.atan2(e1, e2) / (2 * Math.PI)) * 360 + 180;
      const sat = 70,
        light = 50;

      const dim =
        focusSelected && selectedId != null && String(a.id) !== String(selectedId) ? 0.35 : 1.0;

      ctx.save();
      ctx.shadowBlur = px(18);
      ctx.shadowColor = `hsla(${hue},${sat}%,${light}%,${0.35 * dim})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue},${sat}%,${light}%,${0.95 * dim})`;
      ctx.fill();
      ctx.lineWidth = px(2);
      ctx.strokeStyle = `rgba(0,0,0,${0.6 * dim})`;
      ctx.stroke();
      ctx.restore();

      if (selectedId != null && String(a.id) === String(selectedId)) {
        ctx.beginPath();
        ctx.lineWidth = px(3);
        ctx.strokeStyle = "rgba(0,0,0,0.9)";
        ctx.arc(p.x, p.y, r + px(5), 0, Math.PI * 2);
        ctx.stroke();
      }

      // label
      const labelPx = Math.max(11, 12);
      ctx.font = `${labelPx / cam.scale}px ui-sans-serif`;
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(p.x - px(14), p.y + r + px(2) - px(12), px(28), px(16));
      ctx.fillStyle = "rgba(0,0,0,0.85)";
      ctx.fillText(String(a.id), p.x, p.y + r + px(12));
    });

    // screen-space overlay
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "rgba(0,0,0,0.85)";
    ctx.font = "12px ui-sans-serif";
    if (last) ctx.fillText(`${last.action}`, 12, 20);
    if (metrics)
      ctx.fillText(
        `Cohesion: ${metrics.cohesion.toFixed(3)}  |  Polarization: ${metrics.polarization.toFixed(3)}`,
        12,
        38
      );
  };

  return (
    <div className="p-6 grid grid-cols-1 xl:grid-cols-3 gap-6">
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Synthetic Society</span>
            <div className="flex items-center gap-2">
              {/* Planner dropdown */}
              <select
                value={planner}
                onChange={(e) => applyPlanner(e.target.value as Planner)}
                className="border rounded px-2 py-1 text-sm"
                title="Pairing Planner"
              >
                <option value="homophily">Planner: Homophily</option>
                <option value="round_robin">Planner: Round Robin</option>
                <option value="nearest">Planner: Nearest (similar)</option>
                <option value="tension">Planner: Tension (dissimilar)</option>
              </select>

              <Button variant="secondary" onClick={() => zoomButton(0.9)} title="Zoom out">
                −
              </Button>
              <Button variant="secondary" onClick={() => zoomButton(1.1)} title="Zoom in">
                ＋
              </Button>
              <Button variant="secondary" onClick={fitCameraToWorld} title="Fit view">
                Fit
              </Button>
              <Button variant="secondary" onClick={toggleFullscreen} title="Toggle Fullscreen (F)">
                {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            ref={wrapRef}
            className="relative w-full h-[64vh] rounded-lg border bg-card"
            style={{
              boxShadow:
                "0 10px 30px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.7), inset 0 -1px 0 rgba(0,0,0,0.03)",
            }}
          >
            <canvas ref={canvasRef} className="rounded-lg touch-none" />
          </div>

          <div className="flex flex-wrap items-center gap-2 mt-4">
            <Button onClick={step}>Step</Button>
            <Button variant={running ? "destructive" : "default"} onClick={() => setRunning((v) => !v)}>
              {running ? "Stop" : "Run"}
            </Button>
            <Button variant="secondary" onClick={reset}>
              Reset
            </Button>

            <div className="ml-4 flex items-center gap-2">
              <span className="text-sm opacity-70">Speed</span>
              <input
                type="range"
                min={0.5}
                max={12}
                step={0.5}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              />
              <span className="text-sm opacity-70">{speed}x</span>
            </div>

            <div className="ml-auto flex flex-wrap items-center gap-2">
              <Button variant={showHeatmap ? "default" : "secondary"} onClick={() => setShowHeatmap((v) => !v)}>
                Heatmap {showHeatmap ? "✓" : ""}
              </Button>
              <Button variant={showAxes ? "default" : "secondary"} onClick={() => setShowAxes((v) => !v)}>
                Axes {showAxes ? "✓" : ""}
              </Button>
              <Button variant={showRegions ? "default" : "secondary"} onClick={() => setShowRegions((v) => !v)}>
                Regions {showRegions ? "✓" : ""}
              </Button>
              <Button variant={focusSelected ? "default" : "secondary"} onClick={() => setFocusSelected((v) => !v)}>
                Focus Selected {focusSelected ? "✓" : ""}
              </Button>
            </div>
          </div>

          {/* visibility tuning */}
          <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
            <Tuner
              label="Heatmap Intensity"
              value={heatmapIntensity}
              min={0.5}
              max={2.0}
              step={0.1}
              onChange={setHeatmapIntensity}
            />
            <Tuner label="Edge Opacity" value={edgeOpacity} min={0.4} max={1.6} step={0.05} onChange={setEdgeOpacity} />
            <Tuner label="Contrast" value={contrast} min={0.8} max={1.5} step={0.05} onChange={setContrast} />
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => sendEvent("good_news")}>
              Good News
            </Button>
            <Button variant="secondary" onClick={() => sendEvent("market_crash")}>
              Market Crash
            </Button>
            <Button variant="secondary" onClick={() => sendEvent("ad_blitz")}>
              Ad Blitz
            </Button>
            <Button variant="secondary" onClick={() => sendEvent("lockdown")}>
              Lockdown
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Metrics</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3 text-sm">
            {metrics ? (
              <>
                <MetricBox label="Cohesion" value={metrics.cohesion} />
                <MetricBox label="Spread (PC1)" value={metrics.spread_pc1} />
                <MetricBox label="Mean Edge W" value={metrics.mean_edge_weight} />
                <MetricBox label="Polarization" value={metrics.polarization} />
                <div className="col-span-2 p-2 rounded bg-muted/60">
                  <div className="opacity-70 mb-1">Action Counts</div>
                  <div className="grid grid-cols-4 gap-2">
                    {Object.entries(metrics.action_counts).map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between bg-muted px-2 py-1 rounded">
                        <span>{k}</span>
                        <span className="font-mono">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="col-span-2 p-2 rounded bg-muted/60">
                  <div className="opacity-70 mb-1">Trait Means (±sd)</div>
                  <div className="grid grid-cols-4 gap-2">
                    {Object.entries(metrics.trait_means).map(([k, m]) => (
                      <div key={k} className="bg-muted px-2 py-1 rounded">
                        <div className="capitalize">{k}</div>
                        <div className="font-mono">
                          {Number(m).toFixed(2)} ± {Number(metrics.trait_stds[k]).toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div>—</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Agent Personality</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm opacity-70">
              {selectedAgent ? (
                <>
                  Selected: <b>{String(selectedAgent.id)}</b>
                </>
              ) : (
                "Click a node to edit personality."
              )}
            </div>
            {selectedAgent && (
              <>
                {(["optimism", "sociability", "risk", "consumerism"] as const).map((k) => (
                  <div key={k}>
                    <div className="flex items-center justify-between">
                      <label className="text-sm capitalize">{k}</label>
                      <span className="text-xs opacity-70">{Number((traits as any)[k]).toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={-4}
                      max={4}
                      step={0.05}
                      value={(traits as any)[k]}
                      onChange={(e) => setTraits((t) => ({ ...t, [k]: Number(e.target.value) }))}
                      className="w-full"
                    />
                  </div>
                ))}
                <div className="flex gap-2 pt-2">
                  <Button onClick={applyTraits}>Apply</Button>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      const rnd = () => (Math.random() * 3) * (Math.random() < 0.5 ? -1 : 1);
                      setTraits({ optimism: rnd(), sociability: rnd(), risk: rnd(), consumerism: rnd() });
                    }}
                  >
                    Randomize
                  </Button>
                  <Button
                    variant={focusSelected ? "default" : "secondary"}
                    onClick={() => setFocusSelected((v) => !v)}
                    title="Dim other nodes"
                  >
                    {focusSelected ? "Unfocus" : "Focus"}
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[160px] overflow-y-auto font-mono text-xs bg-muted text-foreground p-3 rounded">
              {events?.slice().reverse().map((e, i) => (
                <div key={i} className="border-b border-border py-1">
                  {e}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[180px] overflow-y-auto font-mono text-xs bg-muted text-foreground p-3 rounded">
              {logs.map((l, i) => (
                <div key={i} className="border-b border-border py-1">
                  {l}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  function MetricBox({ label, value }: { label: string; value: number }) {
    return (
      <div className="p-2 rounded bg-muted/60">
        <div className="opacity-70">{label}</div>
        <div className="text-lg">{Number(value).toFixed(3)}</div>
      </div>
    );
  }

  function Tuner({
    label,
    value,
    min,
    max,
    step,
    onChange,
  }: {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (v: number) => void;
  }) {
    return (
      <div className="p-2 rounded bg-muted/60">
        <div className="flex items-center justify-between">
          <span className="opacity-70 text-sm">{label}</span>
          <span className="text-xs font-mono">{value.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
    );
  }

  function zoomButton(factor: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const sx = rect.width / 2;
    const sy = rect.height / 2;
    const cam = cameraRef.current;

    const wx = cam.x + sx / cam.scale;
    const wy = cam.y + sy / cam.scale;
    cam.scale = clamp(cam.scale * factor, MIN_SCALE, MAX_SCALE);
    cam.x = wx - sx / cam.scale;
    cam.y = wy - sy / cam.scale;

    const vw = rect.width / cam.scale;
    const vh = rect.height / cam.scale;
    cam.x = clamp(cam.x, 0, Math.max(0, WORLD_W - vw));
    cam.y = clamp(cam.y, 0, Math.max(0, WORLD_H - vh));
  }
}
