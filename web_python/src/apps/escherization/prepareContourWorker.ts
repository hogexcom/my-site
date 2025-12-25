import { contours as d3Contours } from 'd3-contour'
import simplify from 'simplify-js'

type Vec2 = readonly [number, number]

type WorkerRequest = {
  id: number
  type: 'process'
  width: number
  height: number
  buffer: ArrayBuffer
  options: {
    simplifyRatio: number
    theta1: number
    theta2: number
    alphaThreshold: number
    grayscaleThreshold: number
    maxDimension: number
  }
}

type WorkerResponse =
  | { id: number; type: 'result'; contour: Array<[number, number]>; tranU: [number, number]; tranV: [number, number] }
  | { id: number; type: 'error'; message: string }

function detectNonOpaqueAlpha(data: Uint8ClampedArray): boolean {
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 255) return true
  return false
}

function resampleToMask(
  rgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
  alphaThreshold: number,
  grayscaleThreshold: number,
): Uint8Array {
  const hasAlpha = detectNonOpaqueAlpha(rgba)
  const mask = new Uint8Array(dstW * dstH)
  const scaleX = srcW / dstW
  const scaleY = srcH / dstH

  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y + 0.5) * scaleY))
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x + 0.5) * scaleX))
      const i = (sy * srcW + sx) * 4
      const r = rgba[i + 0]
      const g = rgba[i + 1]
      const b = rgba[i + 2]
      const a = rgba[i + 3]

      if (hasAlpha) {
        mask[y * dstW + x] = a >= alphaThreshold ? 1 : 0
      } else {
        const gray = 0.299 * r + 0.587 * g + 0.114 * b
        mask[y * dstW + x] = gray < grayscaleThreshold ? 1 : 0
      }
    }
  }

  return mask
}

function dilate(mask: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let any = 0
      for (let dy = -1; dy <= 1 && !any; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= height) continue
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx
          if (xx < 0 || xx >= width) continue
          if (mask[yy * width + xx]) {
            any = 1
            break
          }
        }
      }
      out[y * width + x] = any
    }
  }
  return out
}

function erode(mask: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let all = 1
      for (let dy = -1; dy <= 1 && all; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= height) {
          all = 0
          break
        }
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx
          if (xx < 0 || xx >= width) {
            all = 0
            break
          }
          if (!mask[yy * width + xx]) {
            all = 0
            break
          }
        }
      }
      out[y * width + x] = all
    }
  }
  return out
}

function morphologyClose(mask: Uint8Array, width: number, height: number): Uint8Array {
  return erode(dilate(mask, width, height), width, height)
}

function morphologyOpen(mask: Uint8Array, width: number, height: number): Uint8Array {
  return dilate(erode(mask, width, height), width, height)
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

function ensureCcw(points: Vec2[]): Vec2[] {
  return signedArea2(points) < 0 ? points.slice().reverse() : points.slice()
}

function polygonPerimeter(points: Vec2[]): number {
  let p = 0
  const n = points.length
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    p += Math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
  }
  return p
}

function computeParallelogramVectors(contour: Vec2[], theta1: number, theta2: number): readonly [Vec2, Vec2] {
  const area = Math.abs(signedArea2(contour)) / 2
  const thetaDiff = theta2 - theta1
  const sinTheta = Math.abs(Math.sin(thetaDiff))
  if (sinTheta < 1e-10) throw new Error('theta1 and theta2 are nearly parallel')
  const length = Math.sqrt(area / sinTheta)
  const v1: Vec2 = [length * Math.cos(theta1), length * Math.sin(theta1)]
  const v2: Vec2 = [length * Math.cos(theta2), length * Math.sin(theta2)]
  return [v1, v2]
}

function extractLargestContourFromMask(mask: Uint8Array, width: number, height: number): Vec2[] {
  const values = new Array<number>(width * height)
  for (let i = 0; i < values.length; i++) values[i] = mask[i]

  const generator = d3Contours().size([width, height]).thresholds([0.5])
  const geoms = generator(values)
  if (!geoms.length) throw new Error('輪郭が見つかりません')

  const geom = geoms[0]
  if (!geom.coordinates || !geom.coordinates.length) throw new Error('輪郭が見つかりません')

  let bestRing: Vec2[] | null = null
  let bestAbsArea = -1

  for (const polygon of geom.coordinates as Array<Array<Array<[number, number]>>>) {
    if (!polygon.length) continue
    const ring = polygon[0]
    if (ring.length < 3) continue
    const ringVec = ring.map(([x, y]) => [x, y] as const)
    const absArea = Math.abs(signedArea2(ringVec)) / 2
    if (absArea > bestAbsArea) {
      bestAbsArea = absArea
      bestRing = ringVec
    }
  }

  if (!bestRing) throw new Error('輪郭が見つかりません')

  if (bestRing.length >= 2) {
    const a = bestRing[0]
    const b = bestRing[bestRing.length - 1]
    if (a[0] === b[0] && a[1] === b[1]) bestRing = bestRing.slice(0, -1)
  }

  return bestRing
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  if (msg.type !== 'process') return

  const { id, width, height, buffer, options } = msg

  try {
    const rgba = new Uint8ClampedArray(buffer)

    const maxDim = Math.max(width, height)
    const scale = options.maxDimension > 0 && maxDim > options.maxDimension ? options.maxDimension / maxDim : 1
    const procW = Math.max(1, Math.round(width * scale))
    const procH = Math.max(1, Math.round(height * scale))

    const mask0 = resampleToMask(
      rgba,
      width,
      height,
      procW,
      procH,
      options.alphaThreshold,
      options.grayscaleThreshold,
    )

    const cleaned = morphologyOpen(morphologyClose(mask0, procW, procH), procW, procH)
    const ring = extractLargestContourFromMask(cleaned, procW, procH)

    const invScale = scale === 0 ? 1 : 1 / scale
    const scaledBack: Vec2[] = scale === 1 ? ring : ring.map(([x, y]) => [x * invScale, y * invScale] as const)

    const ccw = ensureCcw(scaledBack)

    const perimeter = polygonPerimeter(ccw)
    const epsilon = options.simplifyRatio * perimeter
    const simplifiedObj = simplify(
      ccw.map(([x, y]) => ({ x, y })),
      epsilon,
      true,
    ) as Array<{ x: number; y: number }>
    const simplified: Vec2[] = simplifiedObj.map((p) => [p.x, p.y] as const)
    const finalContour = ensureCcw(simplified.length >= 3 ? simplified : ccw)

    const [tranU, tranV] = computeParallelogramVectors(finalContour, options.theta1, options.theta2)

    const response: WorkerResponse = {
      id,
      type: 'result',
      contour: finalContour.map(([x, y]) => [x, y]),
      tranU: [tranU[0], tranU[1]],
      tranV: [tranV[0], tranV[1]],
    }
    ;(self as any).postMessage(response)
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    const response: WorkerResponse = { id, type: 'error', message }
    ;(self as any).postMessage(response)
  }
}

