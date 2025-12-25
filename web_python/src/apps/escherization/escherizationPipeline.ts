import type { PyodideInterface } from '../../hooks/usePyodide'
import type { PreparedContour } from './prepareContourForEscherization'
import { prepareContourForEscherization } from './prepareContourForEscherization'
import type { Vec2 as V2, TriangulationResult } from './triangulateBoundary'
import { triangulateBoundary } from './triangulateBoundary'
import type { ArapConstraint, ArapMeshDeformResult } from './arapMeshDeform'
import { arapMeshDeform } from './arapMeshDeform'

export type Vec2 = readonly [number, number]

export type TilingPattern = 'P1' | 'P2' | 'P3'

export interface EscherizationPipelineOptions {
  tilingPattern: TilingPattern
  m: number
  n: number
  useResample: boolean
  contourSimplifyRatio?: number
  triangulationMaxArea?: number
  triangulationMaxPoints?: number
  arapIterations?: number
  arapCgMaxIterations?: number
  arapCgTolerance?: number
}

export interface EscherizationPythonResult {
  escherizedBoundary: Vec2[]
  xBest: Vec2[]
  bestIdx: number
  bestError: number
  errors: number[]
  transU: Vec2
  transV: Vec2
}

export interface EscherizationCoreResult {
  contour: PreparedContour
  python: EscherizationPythonResult
  triangulation: TriangulationResult
  arap: ArapMeshDeformResult
}

export interface EscherizationPipelineResult extends EscherizationCoreResult {
  matplotlibPngBase64: string
}

export async function runEscherizationCore(
  imageData: ImageData,
  pyodide: PyodideInterface,
  options: EscherizationPipelineOptions,
): Promise<EscherizationCoreResult> {
  const contour = await prepareContourForEscherization(imageData, { simplifyRatio: options.contourSimplifyRatio })

  const python = await runPythonEscherization(pyodide, contour, options)

  const triangulation = triangulateBoundary(python.xBest as V2[], {
    maxArea: options.triangulationMaxArea,
    maxPoints: options.triangulationMaxPoints,
  })

  const constraints = buildBoundaryConstraints(triangulation.boundaryIndices, python.escherizedBoundary)
  const arap = await arapMeshDeform(triangulation.vertices, triangulation.faces, constraints, {
    iterations: options.arapIterations,
    cgMaxIterations: options.arapCgMaxIterations,
    cgTolerance: options.arapCgTolerance,
  })

  return { contour, python, triangulation, arap }
}

export async function runEscherizationPipeline(
  imageData: ImageData,
  pyodide: PyodideInterface,
  options: EscherizationPipelineOptions,
): Promise<EscherizationPipelineResult> {
  const { contour, python, triangulation, arap } = await runEscherizationCore(imageData, pyodide, options)

  const matplotlibPngBase64 = await renderMatplotlibPng(pyodide, {
    tilingPattern: options.tilingPattern,
    m: options.m,
    n: options.n,
    tranU: python.transU,
    tranV: python.transV,
    bestIdx: python.bestIdx,
    bestError: python.bestError,
    errors: python.errors,
    xBest: python.xBest,
    escherizedBoundary: python.escherizedBoundary,
    vertices: triangulation.vertices,
    faces: triangulation.faces,
    deformedVertices: arap.deformedVertices,
    boundaryIndices: triangulation.boundaryIndices,
  })

  return { contour, python, triangulation, arap, matplotlibPngBase64 }
}

async function runPythonEscherization(
  pyodide: PyodideInterface,
  contour: PreparedContour,
  options: EscherizationPipelineOptions,
): Promise<EscherizationPythonResult> {
  const useResamplePy = options.useResample ? 'True' : 'False'
  const code = `
import numpy as np
from escherize_boundary import escherization

contour = np.array(${JSON.stringify(contour.contour)}, dtype=float)
tran_u = np.array(${JSON.stringify(contour.tranU)}, dtype=float)
tran_v = np.array(${JSON.stringify(contour.tranV)}, dtype=float)

escherized_boundary, X_best, best_idx, best_error, errors = escherization(
    contour,
    ${JSON.stringify(options.tilingPattern)},
    int(${options.m}),
    int(${options.n}),
    ${useResamplePy},
    tran_u,
    tran_v,
)

trans_u_out = None
trans_v_out = None
if ${JSON.stringify(options.tilingPattern)} == "P2":
    import p2
    trans_u_out, trans_v_out = p2.trans_uv(escherized_boundary, int(${options.m}), int(${options.n}))
elif ${JSON.stringify(options.tilingPattern)} == "P3":
    import p3
    trans_u_out, trans_v_out = p3.trans_uv(escherized_boundary, int(${options.m}))
else:
    trans_u_out = tran_u
    trans_v_out = tran_v
`
  await pyodide.runPythonAsync(code)

  const escherizedBoundary = (pyodide.globals.get('escherized_boundary') as any).toJs() as number[][]
  const xBest = (pyodide.globals.get('X_best') as any).toJs() as number[][]
  const bestIdx = toJsNumber(pyodide.globals.get('best_idx'))
  const bestError = toJsNumber(pyodide.globals.get('best_error'))
  const errors = ((pyodide.globals.get('errors') as any).toJs() as number[]) ?? []
  const transU = toVec2(pyodide.globals.get('trans_u_out'))
  const transV = toVec2(pyodide.globals.get('trans_v_out'))

  return {
    escherizedBoundary: escherizedBoundary.map((p) => [p[0], p[1]] as const),
    xBest: xBest.map((p) => [p[0], p[1]] as const),
    bestIdx,
    bestError,
    errors: Array.from(errors),
    transU,
    transV,
  }
}

function buildBoundaryConstraints(boundaryIndices: number[], boundaryPositions: Vec2[]): ArapConstraint[] {
  if (boundaryIndices.length !== boundaryPositions.length) {
    throw new Error(`boundary indices/positions mismatch: ${boundaryIndices.length} vs ${boundaryPositions.length}`)
  }
  const map = new Map<number, Vec2>()
  for (let i = 0; i < boundaryIndices.length; i++) map.set(boundaryIndices[i], boundaryPositions[i])
  return Array.from(map.entries()).map(([index, position]) => ({ index, position }))
}

async function renderMatplotlibPng(
  pyodide: PyodideInterface,
  args: {
    tilingPattern: TilingPattern
    m: number
    n: number
    tranU: Vec2
    tranV: Vec2
    bestIdx: number
    bestError: number
    errors: number[]
    xBest: Vec2[]
    escherizedBoundary: Vec2[]
    vertices: Vec2[]
    faces: Array<readonly [number, number, number]>
    deformedVertices: Vec2[]
    boundaryIndices: number[]
  },
): Promise<string> {
  const code = `
import numpy as np
from visualize_matplotlib import render_escherization_matplotlib_png_base64

X_best = np.array(${JSON.stringify(args.xBest)}, dtype=float)
errors = np.array(${JSON.stringify(args.errors)}, dtype=float)
escherized_boundary = np.array(${JSON.stringify(args.escherizedBoundary)}, dtype=float)
vertices = np.array(${JSON.stringify(args.vertices)}, dtype=float)
faces = np.array(${JSON.stringify(args.faces)}, dtype=int)
deformed_vertices = np.array(${JSON.stringify(args.deformedVertices)}, dtype=float)
boundary_indices = np.array(${JSON.stringify(args.boundaryIndices)}, dtype=int)
tran_u = np.array(${JSON.stringify(args.tranU)}, dtype=float)
tran_v = np.array(${JSON.stringify(args.tranV)}, dtype=float)

matplotlib_png_b64 = render_escherization_matplotlib_png_base64(
    X_best,
    int(${args.bestIdx}),
    errors,
    float(${args.bestError}),
    escherized_boundary,
    vertices,
    faces,
    deformed_vertices,
    boundary_indices,
    ${JSON.stringify(args.tilingPattern)},
    int(${args.m}),
    int(${args.n}),
    tran_u=tran_u,
    tran_v=tran_v,
    dpi=150,
)
`
  await pyodide.runPythonAsync(code)
  const b64Proxy = pyodide.globals.get('matplotlib_png_b64') as any
  return typeof b64Proxy?.toJs === 'function' ? String(b64Proxy.toJs()) : String(b64Proxy)
}

function toJsNumber(v: any): number {
  if (v == null) return NaN
  if (typeof v === 'number') return v
  if (typeof v?.toJs === 'function') return Number(v.toJs())
  return Number(v)
}

function toVec2(v: any): Vec2 {
  if (v == null) return [0, 0]
  if (typeof v?.toJs === 'function') {
    const arr = v.toJs() as number[]
    return [Number(arr[0] ?? 0), Number(arr[1] ?? 0)]
  }
  if (Array.isArray(v)) return [Number(v[0] ?? 0), Number(v[1] ?? 0)]
  return [0, 0]
}
