import * as poly2tri from 'poly2tri'

export type Vec2 = readonly [number, number]

export interface TriangulateBoundaryOptions {
  maxArea?: number
  maxPoints?: number
}

export interface TriangulationResult {
  vertices: Vec2[]
  faces: Array<readonly [number, number, number]>
  boundaryIndices: number[]
}

export function triangulateBoundary(boundary: Vec2[], options: TriangulateBoundaryOptions = {}): TriangulationResult {
  const ring = normalizeRing(boundary)
  if (ring.length < 3) throw new Error('boundary must have at least 3 points')

  const ccwRing = ensureCcw(ring)

  const bbox = boundingBox(ccwRing)
  const defaultMaxArea = (bbox.width * bbox.height) / 500
  const maxArea = options.maxArea ?? defaultMaxArea
  const maxPoints = options.maxPoints ?? 10_000

  const contourPoints = ccwRing.map(([x, y]) => new poly2tri.Point(x, y))
  const sweep = new poly2tri.SweepContext(contourPoints)

  const steiner = generateSteinerPoints(ccwRing, maxArea, maxPoints)
  for (const [x, y] of steiner) sweep.addPoint(new poly2tri.Point(x, y))

  sweep.triangulate()
  const triangles = sweep.getTriangles()

  const pointToIndex = new Map<poly2tri.XY, number>()
  const vertices: Vec2[] = []

  const indexOfPoint = (p: poly2tri.XY): number => {
    const existing = pointToIndex.get(p)
    if (existing !== undefined) return existing
    const idx = vertices.length
    vertices.push([p.x, p.y])
    pointToIndex.set(p, idx)
    return idx
  }

  const faces: Array<readonly [number, number, number]> = []
  for (const tri of triangles) {
    const pts = tri.getPoints()
    const a = indexOfPoint(pts[0])
    const b = indexOfPoint(pts[1])
    const c = indexOfPoint(pts[2])
    faces.push([a, b, c])
  }

  const boundaryIndices = ccwRing.map((p) => findClosestVertexIndex(vertices, p))

  return { vertices, faces, boundaryIndices }
}

function normalizeRing(points: Vec2[]): Vec2[] {
  if (points.length === 0) return []
  const a = points[0]
  const b = points[points.length - 1]
  if (a[0] === b[0] && a[1] === b[1]) return points.slice(0, -1)
  return points.slice()
}

function ensureCcw(points: Vec2[]): Vec2[] {
  return signedArea2(points) < 0 ? points.slice().reverse() : points.slice()
}

function signedArea2(points: Vec2[]): number {
  let a = 0
  const n = points.length
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    a += points[i][0] * points[j][1] - points[j][0] * points[i][1]
  }
  return a
}

function boundingBox(points: Vec2[]): { minX: number; minY: number; maxX: number; maxY: number; width: number; height: number } {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of points) {
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  const width = Math.max(0, maxX - minX)
  const height = Math.max(0, maxY - minY)
  return { minX, minY, maxX, maxY, width, height }
}

function pointInPolygon(p: Vec2, polygon: Vec2[]): boolean {
  const x = p[0]
  const y = p[1]
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0]
    const yi = polygon[i][1]
    const xj = polygon[j][0]
    const yj = polygon[j][1]

    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 0.0) + xi
    if (intersect) inside = !inside
  }
  return inside
}

function generateSteinerPoints(boundary: Vec2[], maxArea: number, maxPoints: number): Vec2[] {
  if (!Number.isFinite(maxArea) || maxArea <= 0) return []

  const bbox = boundingBox(boundary)
  if (bbox.width === 0 || bbox.height === 0) return []

  // Rough spacing so that typical triangle area ~ maxArea.
  // Equilateral: area = sqrt(3)/4 * s^2 => s ~= sqrt(4*area/sqrt(3)).
  const step = Math.max(2, Math.sqrt((4 * maxArea) / Math.sqrt(3)))

  const points: Vec2[] = []
  for (let y = bbox.minY + step / 2; y < bbox.maxY; y += step) {
    for (let x = bbox.minX + step / 2; x < bbox.maxX; x += step) {
      if (points.length >= maxPoints) return points
      const p: Vec2 = [x, y]
      if (pointInPolygon(p, boundary)) points.push(p)
    }
  }

  return points
}

function findClosestVertexIndex(vertices: Vec2[], target: Vec2): number {
  let bestIdx = 0
  let bestD2 = Infinity
  const tx = target[0]
  const ty = target[1]
  for (let i = 0; i < vertices.length; i++) {
    const dx = vertices[i][0] - tx
    const dy = vertices[i][1] - ty
    const d2 = dx * dx + dy * dy
    if (d2 < bestD2) {
      bestD2 = d2
      bestIdx = i
    }
  }
  return bestIdx
}
