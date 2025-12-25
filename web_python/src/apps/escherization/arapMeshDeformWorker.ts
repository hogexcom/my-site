// Ported from https://github.com/PkuCuipy/arap-mesh-deform (ARAP core logic),
// adapted to 2D meshes and to run in a Web Worker.
import numeric from 'numeric'

type WorkerRequest = {
  id: number
  type: 'solve'
  vertices: Float64Array // [x0,y0,x1,y1,...]
  faces: Uint32Array // [a,b,c,...]
  constraints: Uint32Array // [idx0, idx1, ...]
  constraintPositions: Float64Array // [x0,y0,x1,y1,...]
  options: {
    iterations: number
    cgMaxIterations: number
    cgTolerance: number
  }
}

type WorkerResponse =
  | { id: number; type: 'result'; deformedVertices: Float64Array; iterations: number }
  | { id: number; type: 'error'; message: string }

type NeighborTable = Array<number[]>
type OppositeMap = Map<string, number>
type WeightMap = Map<string, number>

function buildNeighborTable(faces: Uint32Array, n: number): NeighborTable {
  const sets: Array<Set<number>> = Array.from({ length: n }, () => new Set())
  for (let t = 0; t < faces.length; t += 3) {
    const a = faces[t + 0]
    const b = faces[t + 1]
    const c = faces[t + 2]
    sets[a].add(b)
    sets[a].add(c)
    sets[b].add(a)
    sets[b].add(c)
    sets[c].add(a)
    sets[c].add(b)
  }
  return sets.map((s) => Array.from(s))
}

function buildOppositeVtxIdOfEdge(faces: Uint32Array): OppositeMap {
  const opp = new Map<string, number>()
  for (let t = 0; t < faces.length; t += 3) {
    const a = faces[t + 0]
    const b = faces[t + 1]
    const c = faces[t + 2]
    opp.set(`${a}-${b}`, c)
    opp.set(`${b}-${c}`, a)
    opp.set(`${c}-${a}`, b)
  }
  return opp
}

function cotangentAtK(vertices: Float64Array, i: number, j: number, k: number): number {
  const kx = vertices[k * 2 + 0]
  const ky = vertices[k * 2 + 1]
  const ix = vertices[i * 2 + 0]
  const iy = vertices[i * 2 + 1]
  const jx = vertices[j * 2 + 0]
  const jy = vertices[j * 2 + 1]

  const v1x = kx - ix
  const v1y = ky - iy
  const v2x = kx - jx
  const v2y = ky - jy

  const dot = v1x * v2x + v1y * v2y
  const cross = Math.abs(v1x * v2y - v1y * v2x)
  if (cross < 1e-12) return 0
  const cot = dot / cross
  return Math.max(cot, 0)
}

function buildWij(vertices: Float64Array, neighborTable: NeighborTable, opposite: OppositeMap): WeightMap {
  const w = new Map<string, number>()
  for (let i = 0; i < neighborTable.length; i++) {
    for (const j of neighborTable[i]) {
      const keyIJ = `${i}-${j}`
      const keyJI = `${j}-${i}`

      let cotAlpha: number | undefined
      const k1 = opposite.get(keyIJ)
      if (k1 !== undefined) cotAlpha = cotangentAtK(vertices, i, j, k1)

      let cotBeta: number | undefined
      const k2 = opposite.get(keyJI)
      if (k2 !== undefined) cotBeta = cotangentAtK(vertices, i, j, k2)

      let wij: number
      if (cotAlpha === undefined) wij = cotBeta ?? 0
      else if (cotBeta === undefined) wij = cotAlpha
      else wij = 0.5 * (cotAlpha + cotBeta)

      w.set(keyIJ, wij)
      w.set(keyJI, wij)
    }
  }
  return w
}

type MatrixLRow = Array<[number, number]> // [j, L_ij], includes diagonal

function buildMatrixL(wij: WeightMap, neighborTable: NeighborTable): Array<MatrixLRow> {
  const n = neighborTable.length
  const L: Array<MatrixLRow> = Array.from({ length: n }, () => [])
  for (let i = 0; i < n; i++) {
    let Lii = 0
    for (const j of neighborTable[i]) {
      const w = wij.get(`${i}-${j}`) ?? 0
      if (!Number.isFinite(w) || w <= 0) continue
      L[i].push([j, -w])
      Lii += w
    }
    L[i].push([i, Lii])
  }
  return L
}

type Rotation2 = readonly [number, number, number, number] // row-major 2x2

function calcRotationMatrices2D(
  p: Float64Array,
  pPrime: Float64Array,
  neighborTable: NeighborTable,
  wij: WeightMap,
): Rotation2[] {
  const n = neighborTable.length
  const R: Rotation2[] = new Array(n)

  for (let i = 0; i < n; i++) {
    let s00 = 0
    let s01 = 0
    let s10 = 0
    let s11 = 0

    const pix = p[i * 2 + 0]
    const piy = p[i * 2 + 1]
    const ppx = pPrime[i * 2 + 0]
    const ppy = pPrime[i * 2 + 1]

    for (const j of neighborTable[i]) {
      const w = wij.get(`${i}-${j}`) ?? 0
      if (!Number.isFinite(w) || w <= 0) continue

      const pjx = p[j * 2 + 0]
      const pjy = p[j * 2 + 1]
      const ppjx = pPrime[j * 2 + 0]
      const ppjy = pPrime[j * 2 + 1]

      const ex = pjx - pix
      const ey = pjy - piy
      const epx = ppjx - ppx
      const epy = ppjy - ppy

      s00 += w * ex * epx
      s01 += w * ex * epy
      s10 += w * ey * epx
      s11 += w * ey * epy
    }

    const Si = [
      [s00, s01],
      [s10, s11],
    ]

    const maxAbs = Math.max(Math.abs(s00), Math.abs(s01), Math.abs(s10), Math.abs(s11), 1)
    Si[0][0] += maxAbs * 1e-6
    Si[1][1] += maxAbs * 1e-6

    const USV = numeric.svd(Si)
    const U: number[][] = USV.U
    const V: number[][] = USV.V

    const detU = numeric.det(U)
    const detV = numeric.det(V)
    if (detU * detV < 0) {
      U[0][1] *= -1
      U[1][1] *= -1
    }

    const UT = numeric.transpose(U)
    const Ri: number[][] = numeric.dot(V, UT)
    const d = numeric.det(Ri)
    if (!Number.isFinite(d)) {
      R[i] = [1, 0, 0, 1]
    } else {
      R[i] = [Ri[0][0], Ri[0][1], Ri[1][0], Ri[1][1]]
    }
  }

  return R
}

function mat2MulVec2(R: Rotation2, x: number, y: number): readonly [number, number] {
  return [R[0] * x + R[1] * y, R[2] * x + R[3] * y]
}

function mat2Add(a: Rotation2, b: Rotation2): Rotation2 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

function mat2Scale(a: Rotation2, s: number): Rotation2 {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

function buildB(
  pOriginal: Float64Array,
  R: Rotation2[],
  neighborTable: NeighborTable,
  wij: WeightMap,
  unknownVertices: number[],
  oriToNew: Int32Array,
  pPrime: Float64Array,
): { bx: Float64Array; by: Float64Array } {
  const m = unknownVertices.length
  const bx = new Float64Array(m)
  const by = new Float64Array(m)

  for (let newI = 0; newI < m; newI++) {
    const i = unknownVertices[newI]
    const Ri = R[i]
    const pix = pOriginal[i * 2 + 0]
    const piy = pOriginal[i * 2 + 1]

    let bix = 0
    let biy = 0
    for (const j of neighborTable[i]) {
      const w = wij.get(`${i}-${j}`) ?? 0
      if (!Number.isFinite(w) || w <= 0) continue
      const Rj = R[j]
      const Rij = mat2Scale(mat2Add(Ri, Rj), 0.5)
      const pjx = pOriginal[j * 2 + 0]
      const pjy = pOriginal[j * 2 + 1]
      const dx = pix - pjx
      const dy = piy - pjy
      const [rx, ry] = mat2MulVec2(Rij, dx, dy)
      bix += w * rx
      biy += w * ry
    }

    bx[newI] = bix
    by[newI] = biy

    // Add known vertices contribution: b -= L_ij * p'_j
    // Here we incorporate them later in buildReducedSystem using L rows.
    // We keep this vector for the "b from rotations" term only.
    void oriToNew
    void pPrime
  }

  return { bx, by }
}

type SparseRows = { cols: Int32Array[]; vals: Float64Array[] }

function buildReducedSystem(
  L: Array<MatrixLRow>,
  unknownVertices: number[],
  oriToNew: Int32Array,
  bx: Float64Array,
  by: Float64Array,
  pPrime: Float64Array,
): SparseRows {
  const m = unknownVertices.length
  const cols: Int32Array[] = new Array(m)
  const vals: Float64Array[] = new Array(m)

  for (let newI = 0; newI < m; newI++) {
    const oriI = unknownVertices[newI]
    const row = L[oriI]

    const rowCols: number[] = []
    const rowVals: number[] = []

    for (const [oriJ, Lij] of row) {
      const newJ = oriToNew[oriJ]
      if (newJ !== -1) {
        rowCols.push(newJ)
        rowVals.push(Lij)
      } else {
        // known vertex: move to RHS
        bx[newI] -= Lij * pPrime[oriJ * 2 + 0]
        by[newI] -= Lij * pPrime[oriJ * 2 + 1]
      }
    }

    cols[newI] = Int32Array.from(rowCols)
    vals[newI] = Float64Array.from(rowVals)
  }

  return { cols, vals }
}

function matVecSparse(rows: SparseRows, x: Float64Array, out: Float64Array): void {
  for (let i = 0; i < rows.cols.length; i++) {
    let sum = 0
    const c = rows.cols[i]
    const v = rows.vals[i]
    for (let k = 0; k < c.length; k++) sum += v[k] * x[c[k]]
    out[i] = sum
  }
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function axpy(alpha: number, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < x.length; i++) y[i] += alpha * x[i]
}

function xpay(alpha: number, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < x.length; i++) y[i] = x[i] + alpha * y[i]
}

function conjugateGradient(A: SparseRows, x: Float64Array, b: Float64Array, maxIter: number, tol: number): Float64Array {
  const n = b.length
  const r = new Float64Array(n)
  const p = new Float64Array(n)
  const Ap = new Float64Array(n)

  matVecSparse(A, x, Ap)
  for (let i = 0; i < n; i++) {
    r[i] = b[i] - Ap[i]
    p[i] = r[i]
  }

  let rsold = dot(r, r)
  const tol2 = tol * tol
  if (rsold < tol2) return x

  for (let iter = 0; iter < maxIter; iter++) {
    matVecSparse(A, p, Ap)
    const denom = dot(p, Ap)
    const alpha = rsold / Math.max(1e-30, denom)
    axpy(alpha, p, x)
    axpy(-alpha, Ap, r)

    const rsnew = dot(r, r)
    if (rsnew < tol2) break

    const beta = rsnew / Math.max(1e-30, rsold)
    xpay(beta, r, p)
    rsold = rsnew
  }

  return x
}

function applyConstraints(pPrime: Float64Array, constraints: Uint32Array, constraintPositions: Float64Array): void {
  for (let i = 0; i < constraints.length; i++) {
    const idx = constraints[i]
    pPrime[idx * 2 + 0] = constraintPositions[i * 2 + 0]
    pPrime[idx * 2 + 1] = constraintPositions[i * 2 + 1]
  }
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  if (msg.type !== 'solve') return

  try {
    const n = msg.vertices.length / 2
    const neighborTable = buildNeighborTable(msg.faces, n)
    const opposite = buildOppositeVtxIdOfEdge(msg.faces)
    const wij = buildWij(msg.vertices, neighborTable, opposite)
    const L = buildMatrixL(wij, neighborTable)

    const isConstrained = new Uint8Array(n)
    for (let i = 0; i < msg.constraints.length; i++) isConstrained[msg.constraints[i]] = 1

    const unknownVertices: number[] = []
    for (let i = 0; i < n; i++) if (!isConstrained[i]) unknownVertices.push(i)

    const oriToNew = new Int32Array(n)
    oriToNew.fill(-1)
    for (let i = 0; i < unknownVertices.length; i++) oriToNew[unknownVertices[i]] = i

    const pOriginal = msg.vertices
    const pPrime = new Float64Array(pOriginal) // init at original
    applyConstraints(pPrime, msg.constraints, msg.constraintPositions)

    for (let it = 0; it < msg.options.iterations; it++) {
      const R = calcRotationMatrices2D(pOriginal, pPrime, neighborTable, wij)
      const { bx, by } = buildB(pOriginal, R, neighborTable, wij, unknownVertices, oriToNew, pPrime)
      const A = buildReducedSystem(L, unknownVertices, oriToNew, bx, by, pPrime)

      const x0 = new Float64Array(unknownVertices.length)
      const y0 = new Float64Array(unknownVertices.length)
      for (let i = 0; i < unknownVertices.length; i++) {
        const oriI = unknownVertices[i]
        x0[i] = pPrime[oriI * 2 + 0]
        y0[i] = pPrime[oriI * 2 + 1]
      }

      conjugateGradient(A, x0, bx, msg.options.cgMaxIterations, msg.options.cgTolerance)
      conjugateGradient(A, y0, by, msg.options.cgMaxIterations, msg.options.cgTolerance)

      for (let i = 0; i < unknownVertices.length; i++) {
        const oriI = unknownVertices[i]
        pPrime[oriI * 2 + 0] = x0[i]
        pPrime[oriI * 2 + 1] = y0[i]
      }

      applyConstraints(pPrime, msg.constraints, msg.constraintPositions)
    }

    const response: WorkerResponse = { id: msg.id, type: 'result', deformedVertices: pPrime, iterations: msg.options.iterations }
    ;(self as any).postMessage(response, [pPrime.buffer])
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    const response: WorkerResponse = { id: msg.id, type: 'error', message }
    ;(self as any).postMessage(response)
  }
}

