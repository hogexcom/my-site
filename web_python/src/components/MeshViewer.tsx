import { useMemo } from 'react'
import * as THREE from 'three'
import { Text } from '@react-three/drei'

export interface MeshData {
  vertices: number[][]
  faces: number[][]
}

interface MeshViewerProps {
  mesh: MeshData
  color: string
  position: [number, number, number]
  label: string
}

export default function MeshViewer({ mesh, color, position, label }: MeshViewerProps) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    
    // 頂点データをFloat32Arrayに変換
    const vertices = new Float32Array(mesh.vertices.flat())
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
    
    // 面データをUint32Arrayに変換
    const indices = new Uint32Array(mesh.faces.flat())
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    
    // 法線を計算
    geo.computeVertexNormals()
    
    return geo
  }, [mesh])

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
        position={[0, -0.8, 0]}
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
