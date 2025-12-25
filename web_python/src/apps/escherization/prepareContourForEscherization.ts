export type Vec2 = readonly [number, number]

export interface PrepareContourOptions {
  simplifyRatio?: number
  theta1?: number
  theta2?: number
  alphaThreshold?: number
  grayscaleThreshold?: number
  maxDimension?: number
  timeoutMs?: number
}

export interface PreparedContour {
  contour: Vec2[]
  imageData: ImageData
  tranU: Vec2
  tranV: Vec2
}

const DEFAULT_OPTIONS: Required<
  Pick<
    PrepareContourOptions,
    'simplifyRatio' | 'theta1' | 'theta2' | 'alphaThreshold' | 'grayscaleThreshold' | 'maxDimension' | 'timeoutMs'
  >
> = {
  simplifyRatio: 0.008,
  theta1: 0,
  theta2: Math.PI / 2,
  alphaThreshold: 128,
  grayscaleThreshold: 250,
  maxDimension: 2048,
  timeoutMs: 90_000,
}

export async function imageDataFromFile(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file)
  return imageDataFromImageBitmap(bitmap)
}

export async function imageDataFromUrl(url: string): Promise<ImageData> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status} ${res.statusText}`)
  const blob = await res.blob()
  const bitmap = await createImageBitmap(blob)
  return imageDataFromImageBitmap(bitmap)
}

export function imageDataFromImageBitmap(bitmap: ImageBitmap): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('Failed to get 2D context')
  ctx.drawImage(bitmap, 0, 0)
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height)
}

type WorkerRequest = {
  id: number
  type: 'process'
  width: number
  height: number
  buffer: ArrayBuffer
  options: Required<
    Pick<PrepareContourOptions, 'simplifyRatio' | 'theta1' | 'theta2' | 'alphaThreshold' | 'grayscaleThreshold' | 'maxDimension'>
  >
}

type WorkerResponse =
  | { id: number; type: 'result'; contour: Array<[number, number]>; tranU: [number, number]; tranV: [number, number] }
  | { id: number; type: 'error'; message: string }

let workerSingleton: Worker | null = null
let nextRequestId = 1
const pending = new Map<number, { resolve: (r: WorkerResponse) => void; reject: (e: unknown) => void }>()

function getWorker(): Worker {
  if (workerSingleton) return workerSingleton

  const worker = new Worker(new URL('./prepareContourWorker.ts', import.meta.url), { type: 'module' })

  worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
    const msg = e.data
    const handlers = pending.get(msg.id)
    if (!handlers) return
    pending.delete(msg.id)
    if (msg.type === 'error') handlers.reject(new Error(msg.message))
    else handlers.resolve(msg)
  }

  worker.onerror = (e) => {
    console.error('prepareContourWorker error', e)
  }

  workerSingleton = worker
  return workerSingleton
}

function resetWorker(): void {
  if (workerSingleton) workerSingleton.terminate()
  workerSingleton = null
}

export async function prepareContourForEscherization(
  imageData: ImageData,
  options: PrepareContourOptions = {},
): Promise<PreparedContour> {
  const opts = { ...DEFAULT_OPTIONS, ...options }

  const requestId = nextRequestId++
  const worker = getWorker()

  const width = imageData.width
  const height = imageData.height

  const copied = new Uint8ClampedArray(imageData.data)
  const buffer = copied.buffer

  const request: WorkerRequest = {
    id: requestId,
    type: 'process',
    width,
    height,
    buffer,
    options: {
      simplifyRatio: opts.simplifyRatio,
      theta1: opts.theta1,
      theta2: opts.theta2,
      alphaThreshold: opts.alphaThreshold,
      grayscaleThreshold: opts.grayscaleThreshold,
      maxDimension: opts.maxDimension,
    },
  }

  const responsePromise = new Promise<WorkerResponse>((resolve, reject) => {
    pending.set(requestId, { resolve, reject })
    worker.postMessage(request, [buffer])
  })

  const timeoutPromise = new Promise<never>((_, reject) => {
    const t = window.setTimeout(() => {
      resetWorker()
      reject(new Error(`prepareContourForEscherization timed out after ${opts.timeoutMs}ms`))
    }, opts.timeoutMs)
    responsePromise.finally(() => window.clearTimeout(t))
  })

  const response = (await Promise.race([responsePromise, timeoutPromise])) as WorkerResponse
  if (response.type !== 'result') throw new Error('Unexpected worker response')

  const contour = ensureCcw(response.contour)
  return { contour, imageData, tranU: response.tranU, tranV: response.tranV }
}

function ensureCcw(points: Vec2[]): Vec2[] {
  const area2 = signedArea2(points)
  return area2 < 0 ? points.slice().reverse() : points.slice()
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

