import { useMemo } from 'react'
import * as THREE from 'three'
import { Text } from '@react-three/drei'

export interface MeshData {
  vertices: number[][]
  faces: number[][]
}

export interface BoundingInfo {
  center: [number, number, number]
  size: number
  minY: number  // 底面のY座標
}

// バウンディングボックスを計算
export function computeBoundingInfo(mesh: MeshData): BoundingInfo {
  const vertices = mesh.vertices
  if (vertices.length === 0) {
    return { center: [0, 0, 0], size: 1, minY: 0 }
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

  for (const [x, y, z] of vertices) {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    minZ = Math.min(minZ, z)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
    maxZ = Math.max(maxZ, z)
  }

  const center: [number, number, number] = [
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2
  ]

  const sizeX = maxX - minX
  const sizeY = maxY - minY
  const sizeZ = maxZ - minZ
  const size = Math.max(sizeX, sizeY, sizeZ)

  return { center, size, minY }
}

interface MeshViewerProps {
  mesh: MeshData
  color: string
  position?: [number, number, number]
  label: string
  boundingInfo?: BoundingInfo
}

export default function MeshViewer({ mesh, color, position = [0, 0, 0], label, boundingInfo }: MeshViewerProps) {
  const { geometry, labelY } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    
    // バウンディング情報を計算または使用
    const info = boundingInfo || computeBoundingInfo(mesh)
    const [cx, cy, cz] = info.center
    const scale = info.size > 0 ? 1.5 / info.size : 1
    
    // 底面がグリッド（Y=0）に接するようにオフセットを計算
    const yOffset = (cy - info.minY) * scale
    
    // 頂点を正規化（中心に移動してスケール調整、底面をグリッドに接する）
    const normalizedVertices = new Float32Array(mesh.vertices.length * 3)
    for (let i = 0; i < mesh.vertices.length; i++) {
      const [x, y, z] = mesh.vertices[i]
      normalizedVertices[i * 3] = (x - cx) * scale
      normalizedVertices[i * 3 + 1] = (y - cy) * scale + yOffset
      normalizedVertices[i * 3 + 2] = (z - cz) * scale
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(normalizedVertices, 3))
    
    // 面データをUint32Arrayに変換
    const indices = new Uint32Array(mesh.faces.flat())
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    
    // 法線を計算
    geo.computeVertexNormals()
    
    // ラベルのY位置を計算（バウンディングボックスの上）
    geo.computeBoundingBox()
    const labelY = geo.boundingBox ? geo.boundingBox.max.y + 0.15 : 0.8
    
    return { geometry: geo, labelY }
  }, [mesh, boundingInfo])

  return (
    <group position={position}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color={color}
          side={THREE.DoubleSide}
          flatShading={false}
        />
      </mesh>
      <Text
        position={[0, labelY, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  )
}
