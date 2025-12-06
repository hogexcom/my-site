import { useMemo } from 'react'
import * as THREE from 'three'
import { Text } from '@react-three/drei'
import { type MeshData, type BoundingInfo, computeBoundingInfo } from '../../components/MeshViewer'

interface EigenvectorViewerProps {
  mesh: MeshData
  eigenvector: number[] | null  // 頂点ごとの固有ベクトルの値
  position?: [number, number, number]
  label: string
  boundingInfo?: BoundingInfo
}

// 値を色に変換（青→白→赤のカラーマップ）
function valueToColor(value: number, minVal: number, maxVal: number): [number, number, number] {
  // -1 to 1 の範囲に正規化
  const range = Math.max(Math.abs(minVal), Math.abs(maxVal))
  const normalized = range > 0 ? value / range : 0
  
  // 青(-1) → 白(0) → 赤(1) のカラーマップ
  if (normalized < 0) {
    // 青から白へ
    const t = 1 + normalized  // 0 to 1
    return [t, t, 1]  // (0,0,1) → (1,1,1)
  } else {
    // 白から赤へ
    const t = 1 - normalized  // 1 to 0
    return [1, t, t]  // (1,1,1) → (1,0,0)
  }
}

export default function EigenvectorViewer({ 
  mesh, 
  eigenvector, 
  position = [0, 0, 0], 
  label, 
  boundingInfo 
}: EigenvectorViewerProps) {
  const { geometry, labelY } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    
    // バウンディング情報を計算または使用
    const info = boundingInfo || computeBoundingInfo(mesh)
    const [cx, cy, cz] = info.center
    const scale = info.size > 0 ? 1.5 / info.size : 1
    
    // 底面がグリッド（Y=0）に接するようにオフセットを計算
    const yOffset = (cy - info.minY) * scale
    
    // 面ごとに頂点を展開（頂点カラーをスムーズに補間するため）
    const positions: number[] = []
    const colors: number[] = []
    
    // 固有ベクトルの最小・最大値を計算
    let minVal = 0, maxVal = 0
    if (eigenvector && eigenvector.length > 0) {
      minVal = Math.min(...eigenvector)
      maxVal = Math.max(...eigenvector)
    }
    
    for (const face of mesh.faces) {
      for (const vertexIndex of face) {
        const [x, y, z] = mesh.vertices[vertexIndex]
        // 正規化された頂点座標（底面をグリッドに接する）
        positions.push(
          (x - cx) * scale,
          (y - cy) * scale + yOffset,
          (z - cz) * scale
        )
        
        // 頂点カラー
        if (eigenvector && eigenvector.length > vertexIndex) {
          const [r, g, b] = valueToColor(eigenvector[vertexIndex], minVal, maxVal)
          colors.push(r, g, b)
        } else {
          colors.push(0.5, 0.5, 0.5)  // グレー（データなし）
        }
      }
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3))
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3))
    
    // 法線を計算
    geo.computeVertexNormals()
    
    // ラベルのY位置を計算
    geo.computeBoundingBox()
    const labelY = geo.boundingBox ? geo.boundingBox.max.y + 0.15 : 0.8
    
    return { geometry: geo, labelY }
  }, [mesh, eigenvector, boundingInfo])

  return (
    <group position={position}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
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
