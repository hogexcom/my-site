import type { TilingPattern, Vec2 } from './escherizationPipeline'

export type Triangle = readonly [number, number, number]

export interface TilingBuildInput {
  tilingPattern: TilingPattern
  m: number
  n: number
  nx: number
  ny: number
  deformedVertices: Vec2[]
  escherizedBoundary: Vec2[]
  transU: Vec2
  transV: Vec2
}

export interface TilingDrawInput {
  ctx: CanvasRenderingContext2D
  image: CanvasImageSource
  sourceVertices: Vec2[]
  faces: Triangle[]
  tiles: Vec2[][]
  width: number
  height: number
  view?: {
    zoom: number
    panX: number
    panY: number
  }
}

export function buildTilingTiles(input: TilingBuildInput): Vec2[][] {
  const { tilingPattern, m, n, nx, ny, deformedVertices, escherizedBoundary, transU, transV } = input

  const baseTiles =
    tilingPattern === 'P2'
      ? buildP2BaseTiles(deformedVertices, escherizedBoundary, m, n)
      : tilingPattern === 'P3'
        ? buildP3BaseTiles(deformedVertices, escherizedBoundary, m)
        : [deformedVertices]

  const tiles: Vec2[][] = []
  const countX =
    tilingPattern === 'P2' ? Math.max(1, Math.floor(nx / 2)) : Math.max(1, Math.floor(nx))
  const countY =
    tilingPattern === 'P2' ? Math.max(1, Math.floor(ny / 2)) : Math.max(1, Math.floor(ny))
  for (let i = 0; i < countX; i++) {
    for (let j = 0; j < countY; j++) {
      const translation: Vec2 = [transU[0] * i + transV[0] * j, transU[1] * i + transV[1] * j]
      for (const tile of baseTiles) {
        tiles.push(translateVertices(tile, translation))
      }
    }
  }

  return tiles
}

export function drawTiledMesh({ ctx, image, sourceVertices, faces, tiles, width, height, view }: TilingDrawInput): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, width, height)
  drawBackground(ctx, width, height)

  const bounds = computeBounds(tiles)
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
  const padding = 24
  const availableWidth = Math.max(1, width - padding * 2)
  const availableHeight = Math.max(1, height - padding * 2)
  const baseScale = Math.max(availableWidth / spanX, availableHeight / spanY)
  const zoom = view?.zoom ?? 1
  const centerX = width / 2
  const centerY = height / 2
  const boundsCenterX = (bounds.minX + bounds.maxX) / 2
  const boundsCenterY = (bounds.minY + bounds.maxY) / 2
  const baseOffsetX = centerX - boundsCenterX * baseScale
  const baseOffsetY = centerY - boundsCenterY * baseScale
  const scale = baseScale * zoom
  const offsetX = baseOffsetX * zoom + (view?.panX ?? 0)
  const offsetY = baseOffsetY * zoom + (view?.panY ?? 0)

  const toScreen = (p: Vec2): Vec2 => [p[0] * scale + offsetX, p[1] * scale + offsetY]

  ctx.imageSmoothingEnabled = true

  for (const tile of tiles) {
    for (const face of faces) {
      const s0 = sourceVertices[face[0]]
      const s1 = sourceVertices[face[1]]
      const s2 = sourceVertices[face[2]]
      const d0 = toScreen(tile[face[0]])
      const d1 = toScreen(tile[face[1]])
      const d2 = toScreen(tile[face[2]])
      drawImageTriangle(ctx, image, s0, s1, s2, d0, d1, d2)
    }
  }
}

function drawBackground(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  const gradient = ctx.createLinearGradient(0, 0, width, height)
  gradient.addColorStop(0, '#f8fafc')
  gradient.addColorStop(1, '#e2e8f0')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, width, height)

  ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)'
  ctx.lineWidth = 1
  const step = 48
  for (let x = 0; x <= width; x += step) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }
  for (let y = 0; y <= height; y += step) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }
}

function computeBounds(tiles: Vec2[][]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const tile of tiles) {
    for (const [x, y] of tile) {
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
    }
  }
  if (!Number.isFinite(minX)) {
    minX = 0
    minY = 0
    maxX = 1
    maxY = 1
  }
  return { minX, minY, maxX, maxY }
}

function buildP2BaseTiles(vertices: Vec2[], boundary: Vec2[], m: number, n: number): Vec2[][] {
  const { center0, center1, center2 } = p2Centers(m, n)
  const center1Pos = boundary[center1]
  const center2Pos = boundary[center2]
  const center0Pos = boundary[center0]

  const angle = Math.PI
  const tileA = rotateVertices(vertices, center1Pos, angle)
  const tileB = rotateVertices(vertices, center2Pos, angle)

  const tileACenter0 = rotatePoint(center0Pos, center1Pos, angle)
  const tileC = rotateVertices(tileA, tileACenter0, angle)

  return [vertices, tileA, tileB, tileC]
}

function buildP3BaseTiles(vertices: Vec2[], boundary: Vec2[], m: number): Vec2[][] {
  const rotIndices = p3RotationIndices(m)
  const tile1 = rotateVertices(vertices, boundary[rotIndices[0]], (-2 * Math.PI) / 3)
  const tile2 = rotateVertices(vertices, boundary[rotIndices[1]], (2 * Math.PI) / 3)
  return [vertices, tile1, tile2]
}

function translateVertices(vertices: Vec2[], translation: Vec2): Vec2[] {
  return vertices.map(([x, y]) => [x + translation[0], y + translation[1]] as const)
}

function rotateVertices(vertices: Vec2[], center: Vec2, angle: number): Vec2[] {
  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const cx = center[0]
  const cy = center[1]
  return vertices.map(([x, y]) => {
    const dx = x - cx
    const dy = y - cy
    return [cosA * dx - sinA * dy + cx, sinA * dx + cosA * dy + cy] as const
  })
}

function rotatePoint(point: Vec2, center: Vec2, angle: number): Vec2 {
  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const dx = point[0] - center[0]
  const dy = point[1] - center[1]
  return [cosA * dx - sinA * dy + center[0], sinA * dx + cosA * dy + center[1]]
}

function p2Centers(m: number, n: number): { center0: number; center1: number; center2: number; center3: number } {
  const edge0 = Array.from({ length: m }, (_, i) => i)
  const edge1 = Array.from({ length: n }, (_, i) => m - 1 + i)
  const edge2 = Array.from({ length: m }, (_, i) => m + n - 2 + i)
  const edge3 = Array.from({ length: n }, (_, i) => 2 * m + n - 3 + i)
  edge3[edge3.length - 1] = 0
  return {
    center0: edge0[Math.floor(edge0.length / 2)],
    center1: edge1[Math.floor(edge1.length / 2)],
    center2: edge2[Math.floor(edge2.length / 2)],
    center3: edge3[Math.floor(edge3.length / 2)],
  }
}

function p3RotationIndices(m: number): [number, number, number] {
  const edge0 = Array.from({ length: m }, (_, i) => i)
  let last = m - 1
  last = last + m - 1
  const edge2 = Array.from({ length: m }, (_, i) => last + i)
  last = last + m - 1
  last = last + m - 1
  const edge4 = Array.from({ length: m }, (_, i) => last + i)
  last = last + m - 1
  const edge5 = Array.from({ length: m }, (_, i) => last + i)
  edge5[edge5.length - 1] = 0
  return [edge0[edge0.length - 1], edge2[edge2.length - 1], edge4[edge4.length - 1]]
}

function drawImageTriangle(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  s0: Vec2,
  s1: Vec2,
  s2: Vec2,
  d0: Vec2,
  d1: Vec2,
  d2: Vec2,
): void {
  const denom = s0[0] * (s1[1] - s2[1]) + s1[0] * (s2[1] - s0[1]) + s2[0] * (s0[1] - s1[1])
  if (Math.abs(denom) < 1e-8) return

  const a =
    (d0[0] * (s1[1] - s2[1]) + d1[0] * (s2[1] - s0[1]) + d2[0] * (s0[1] - s1[1])) / denom
  const b =
    (d0[1] * (s1[1] - s2[1]) + d1[1] * (s2[1] - s0[1]) + d2[1] * (s0[1] - s1[1])) / denom
  const c =
    (d0[0] * (s2[0] - s1[0]) + d1[0] * (s0[0] - s2[0]) + d2[0] * (s1[0] - s0[0])) / denom
  const d =
    (d0[1] * (s2[0] - s1[0]) + d1[1] * (s0[0] - s2[0]) + d2[1] * (s1[0] - s0[0])) / denom
  const e =
    (d0[0] * (s1[0] * s2[1] - s2[0] * s1[1]) +
      d1[0] * (s2[0] * s0[1] - s0[0] * s2[1]) +
      d2[0] * (s0[0] * s1[1] - s1[0] * s0[1])) /
    denom
  const f =
    (d0[1] * (s1[0] * s2[1] - s2[0] * s1[1]) +
      d1[1] * (s2[0] * s0[1] - s0[0] * s2[1]) +
      d2[1] * (s0[0] * s1[1] - s1[0] * s0[1])) /
    denom

  ctx.save()
  ctx.beginPath()
  ctx.moveTo(d0[0], d0[1])
  ctx.lineTo(d1[0], d1[1])
  ctx.lineTo(d2[0], d2[1])
  ctx.closePath()
  ctx.clip()
  ctx.setTransform(a, b, c, d, e, f)
  ctx.drawImage(image, 0, 0)
  ctx.restore()
}
