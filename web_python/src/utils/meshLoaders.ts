import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import type { MeshData } from '../components/MeshViewer'

// サポートする形式
export const SUPPORTED_FORMATS = {
  '.obj': 'Wavefront OBJ',
  '.stl': 'STL (Stereolithography)',
  '.ply': 'PLY (Polygon File Format)',
  '.gltf': 'glTF',
  '.glb': 'glTF Binary',
  '.fbx': 'Autodesk FBX',
}

export const SUPPORTED_EXTENSIONS = Object.keys(SUPPORTED_FORMATS).join(',')

// ファイル拡張子を取得
function getExtension(filename: string): string {
  const ext = filename.toLowerCase().match(/\.[^.]+$/)
  return ext ? ext[0] : ''
}

// THREE.Geometryまたは THREE.BufferGeometry からMeshDataを抽出
function extractMeshData(geometry: THREE.BufferGeometry): MeshData {
  const vertices: number[][] = []
  const faces: number[][] = []
  
  // インデックスがない場合は作成
  if (!geometry.index) {
    const posAttr = geometry.getAttribute('position')
    const indices: number[] = []
    for (let i = 0; i < posAttr.count; i++) {
      indices.push(i)
    }
    geometry.setIndex(indices)
  }
  
  // 頂点座標を取得
  const positionAttr = geometry.getAttribute('position')
  for (let i = 0; i < positionAttr.count; i++) {
    vertices.push([
      positionAttr.getX(i),
      positionAttr.getY(i),
      positionAttr.getZ(i)
    ])
  }
  
  // 面（三角形）を取得
  const index = geometry.index
  if (index) {
    for (let i = 0; i < index.count; i += 3) {
      faces.push([
        index.getX(i),
        index.getX(i + 1),
        index.getX(i + 2)
      ])
    }
  }
  
  return { vertices, faces }
}

// 複数のメッシュをマージする
function mergeGeometries(object: THREE.Object3D): THREE.BufferGeometry | null {
  const geometries: THREE.BufferGeometry[] = []
  
  object.traverse((child) => {
    if (child instanceof THREE.Mesh && child.geometry) {
      const geo = child.geometry.clone()
      
      // ワールド変換を適用
      child.updateWorldMatrix(true, false)
      geo.applyMatrix4(child.matrixWorld)
      
      // 頂点のみの geometry を作成（法線やUVは不要）
      const posAttr = geo.getAttribute('position')
      const newGeo = new THREE.BufferGeometry()
      newGeo.setAttribute('position', posAttr)
      if (geo.index) {
        newGeo.setIndex(geo.index)
      }
      
      geometries.push(newGeo)
    }
  })
  
  if (geometries.length === 0) return null
  if (geometries.length === 1) return geometries[0]
  
  // 複数のジオメトリをマージ
  const merged = new THREE.BufferGeometry()
  const positions: number[] = []
  const indices: number[] = []
  let vertexOffset = 0
  
  for (const geo of geometries) {
    const posAttr = geo.getAttribute('position')
    
    // 頂点を追加
    for (let i = 0; i < posAttr.count; i++) {
      positions.push(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i))
    }
    
    // インデックスを追加（オフセット付き）
    if (geo.index) {
      for (let i = 0; i < geo.index.count; i++) {
        indices.push(geo.index.getX(i) + vertexOffset)
      }
    } else {
      // インデックスがない場合は連番
      for (let i = 0; i < posAttr.count; i++) {
        indices.push(i + vertexOffset)
      }
    }
    
    vertexOffset += posAttr.count
  }
  
  merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  merged.setIndex(indices)
  
  return merged
}

// OBJファイルをパース（手動パースで頂点共有を維持）
async function loadOBJ(content: string): Promise<MeshData> {
  const vertices: number[][] = []
  const faces: number[][] = []
  
  const lines = content.split('\n')
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    
    const parts = trimmed.split(/\s+/)
    
    if (parts[0] === 'v') {
      vertices.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3])
      ])
    } else if (parts[0] === 'f') {
      // 三角形のみサポート（最初の3頂点を使用）
      const faceIndices = parts.slice(1, 4).map(p => {
        const idx = parseInt(p.split('/')[0]) - 1 // OBJは1-indexed
        return idx
      })
      if (faceIndices.length === 3) {
        faces.push(faceIndices)
      }
    }
  }
  
  if (vertices.length === 0 || faces.length === 0) {
    throw new Error('No valid mesh data found in OBJ file')
  }
  
  return { vertices, faces }
}

// STLファイルをパース
async function loadSTL(buffer: ArrayBuffer): Promise<MeshData> {
  const loader = new STLLoader()
  const geometry = loader.parse(buffer)
  
  return extractMeshData(geometry)
}

// PLYファイルをパース
async function loadPLY(buffer: ArrayBuffer): Promise<MeshData> {
  const loader = new PLYLoader()
  const geometry = loader.parse(buffer)
  
  return extractMeshData(geometry)
}

// glTF/GLBファイルをパース
async function loadGLTF(buffer: ArrayBuffer): Promise<MeshData> {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader()
    
    // ArrayBufferをBlobに変換してURL作成
    const blob = new Blob([buffer])
    const url = URL.createObjectURL(blob)
    
    loader.load(
      url,
      (gltf) => {
        URL.revokeObjectURL(url)
        
        const geometry = mergeGeometries(gltf.scene)
        if (!geometry) {
          reject(new Error('No mesh found in glTF file'))
          return
        }
        
        resolve(extractMeshData(geometry))
      },
      undefined,
      (error) => {
        URL.revokeObjectURL(url)
        reject(error)
      }
    )
  })
}

// FBXファイルをパース
async function loadFBX(buffer: ArrayBuffer): Promise<MeshData> {
  return new Promise((resolve, reject) => {
    const loader = new FBXLoader()
    
    // ArrayBufferをBlobに変換してURL作成
    const blob = new Blob([buffer])
    const url = URL.createObjectURL(blob)
    
    loader.load(
      url,
      (fbx) => {
        URL.revokeObjectURL(url)
        
        const geometry = mergeGeometries(fbx)
        if (!geometry) {
          reject(new Error('No mesh found in FBX file'))
          return
        }
        
        resolve(extractMeshData(geometry))
      },
      undefined,
      (error) => {
        URL.revokeObjectURL(url)
        reject(error)
      }
    )
  })
}

// メインのローダー関数
export async function loadMeshFile(file: File): Promise<MeshData> {
  const ext = getExtension(file.name)
  
  switch (ext) {
    case '.obj': {
      const text = await file.text()
      return loadOBJ(text)
    }
    
    case '.stl': {
      const buffer = await file.arrayBuffer()
      return loadSTL(buffer)
    }
    
    case '.ply': {
      const buffer = await file.arrayBuffer()
      return loadPLY(buffer)
    }
    
    case '.gltf':
    case '.glb': {
      const buffer = await file.arrayBuffer()
      return loadGLTF(buffer)
    }
    
    case '.fbx': {
      const buffer = await file.arrayBuffer()
      return loadFBX(buffer)
    }
    
    default:
      throw new Error(`Unsupported file format: ${ext}`)
  }
}

// URLからメッシュを読み込む（デフォルトモデル用）
export async function loadMeshFromURL(url: string): Promise<MeshData> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch: ${url}`)
  }
  
  const ext = getExtension(url)
  
  switch (ext) {
    case '.obj': {
      const text = await response.text()
      return loadOBJ(text)
    }
    
    case '.stl': {
      const buffer = await response.arrayBuffer()
      return loadSTL(buffer)
    }
    
    case '.ply': {
      const buffer = await response.arrayBuffer()
      return loadPLY(buffer)
    }
    
    case '.gltf':
    case '.glb': {
      const buffer = await response.arrayBuffer()
      return loadGLTF(buffer)
    }
    
    case '.fbx': {
      const buffer = await response.arrayBuffer()
      return loadFBX(buffer)
    }
    
    default:
      throw new Error(`Unsupported file format: ${ext}`)
  }
}
