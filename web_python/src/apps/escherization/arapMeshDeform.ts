export type Vec2 = readonly [number, number]

export interface ArapConstraint {
  index: number
  position: Vec2
}

export interface ArapMeshDeformOptions {
  iterations?: number
  cgMaxIterations?: number
  cgTolerance?: number
  timeoutMs?: number
}

export interface ArapMeshDeformResult {
  deformedVertices: Vec2[]
  iterations: number
}

type WorkerRequest = {
  id: number
  type: 'solve'
  vertices: Float64Array // [x0,y0,x1,y1,...]
  faces: Uint32Array // [a,b,c,...]
  constraints: Uint32Array // [idx0, idx1, ...]
  constraintPositions: Float64Array // [x0,y0,x1,y1,...]
  options: Required<Pick<ArapMeshDeformOptions, 'iterations' | 'cgMaxIterations' | 'cgTolerance'>>
}

type WorkerResponse =
  | { id: number; type: 'result'; deformedVertices: Float64Array; iterations: number }
  | { id: number; type: 'error'; message: string }

const DEFAULT_OPTIONS: Required<Pick<ArapMeshDeformOptions, 'iterations' | 'cgMaxIterations' | 'cgTolerance' | 'timeoutMs'>> = {
  iterations: 5,
  cgMaxIterations: 400,
  cgTolerance: 1e-6,
  timeoutMs: 60_000,
}

let workerSingleton: Worker | null = null
let nextRequestId = 1
const pending = new Map<number, { resolve: (r: WorkerResponse) => void; reject: (e: unknown) => void }>()

function getWorker(): Worker {
  if (workerSingleton) return workerSingleton
  const worker = new Worker(new URL('./arapMeshDeformWorker.ts', import.meta.url), { type: 'module' })

  worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
    const msg = e.data
    const handlers = pending.get(msg.id)
    if (!handlers) return
    pending.delete(msg.id)
    if (msg.type === 'error') handlers.reject(new Error(msg.message))
    else handlers.resolve(msg)
  }

  worker.onerror = (e) => {
    console.error('arapMeshDeformWorker error', e)
  }

  workerSingleton = worker
  return workerSingleton
}

function resetWorker(): void {
  if (workerSingleton) workerSingleton.terminate()
  workerSingleton = null
}

export async function arapMeshDeform(
  verticesOriginal: Vec2[],
  faces: Array<readonly [number, number, number]>,
  constraints: ArapConstraint[],
  options: ArapMeshDeformOptions = {},
): Promise<ArapMeshDeformResult> {
  const opts = { ...DEFAULT_OPTIONS, ...options }
  if (faces.length === 0) throw new Error('faces is empty')
  if (constraints.length < 2) throw new Error('at least 2 constraints are required')

  const requestId = nextRequestId++
  const worker = getWorker()

  const v = new Float64Array(verticesOriginal.length * 2)
  for (let i = 0; i < verticesOriginal.length; i++) {
    v[i * 2 + 0] = verticesOriginal[i][0]
    v[i * 2 + 1] = verticesOriginal[i][1]
  }

  const f = new Uint32Array(faces.length * 3)
  for (let i = 0; i < faces.length; i++) {
    f[i * 3 + 0] = faces[i][0]
    f[i * 3 + 1] = faces[i][1]
    f[i * 3 + 2] = faces[i][2]
  }

  const c = new Uint32Array(constraints.length)
  const cp = new Float64Array(constraints.length * 2)
  for (let i = 0; i < constraints.length; i++) {
    c[i] = constraints[i].index
    cp[i * 2 + 0] = constraints[i].position[0]
    cp[i * 2 + 1] = constraints[i].position[1]
  }

  const request: WorkerRequest = {
    id: requestId,
    type: 'solve',
    vertices: v,
    faces: f,
    constraints: c,
    constraintPositions: cp,
    options: {
      iterations: opts.iterations,
      cgMaxIterations: opts.cgMaxIterations,
      cgTolerance: opts.cgTolerance,
    },
  }

  const responsePromise = new Promise<WorkerResponse>((resolve, reject) => {
    pending.set(requestId, { resolve, reject })
    worker.postMessage(request, [v.buffer, f.buffer, c.buffer, cp.buffer])
  })

  const timeoutPromise = new Promise<never>((_, reject) => {
    const t = window.setTimeout(() => {
      resetWorker()
      reject(new Error(`arapMeshDeform timed out after ${opts.timeoutMs}ms`))
    }, opts.timeoutMs)
    responsePromise.finally(() => window.clearTimeout(t))
  })

  const response = (await Promise.race([responsePromise, timeoutPromise])) as WorkerResponse
  if (response.type !== 'result') throw new Error('Unexpected worker response')

  const out: Vec2[] = []
  const dv = response.deformedVertices
  for (let i = 0; i < dv.length / 2; i++) out.push([dv[i * 2 + 0], dv[i * 2 + 1]])
  return { deformedVertices: out, iterations: response.iterations }
}

