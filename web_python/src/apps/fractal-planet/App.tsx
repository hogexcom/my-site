import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { usePyodide } from '../../hooks/usePyodide';
import './App.css';

interface PlanetTexture {
  faultCount: number;
  texture: THREE.DataTexture;
  canvas2D: HTMLCanvasElement;
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    planet: THREE.Mesh;
    stars: THREE.Points;
  } | null>(null);

  const [generating, setGenerating] = useState(false);
  const [textures, setTextures] = useState<PlanetTexture[]>([]);
  const [selectedMap, setSelectedMap] = useState<PlanetTexture | null>(null);
  const { pyodide, isReady: pyodideLoading, error: pyodideError } = usePyodide(['fractal_planet.py']);

  // Initialize Three.js scene
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 10;

    // Starfield
    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 5000;
    const positions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      // Random positions in a sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 50 + Math.random() * 50;
      
      positions[i] = r * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = r * Math.cos(phi);
    }
    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.1,
      transparent: true,
      opacity: 0.8,
    });
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    // Sun (directional light with glow)
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.5);
    sunLight.position.set(10, 5, 5);
    scene.add(sunLight);

    // Add sun sphere
    const sunGeometry = new THREE.SphereGeometry(1, 32, 32);
    const sunMaterial = new THREE.MeshBasicMaterial({
      color: 0xffff00,
    });
    const sun = new THREE.Mesh(sunGeometry, sunMaterial);
    sun.position.copy(sunLight.position);
    scene.add(sun);

    // Add sun glow
    const glowGeometry = new THREE.SphereGeometry(1.2, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: 0xffaa00,
      transparent: true,
      opacity: 0.3,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    glow.position.copy(sunLight.position);
    scene.add(glow);

    // Ambient light (brighter to illuminate dark side)
    const ambientLight = new THREE.AmbientLight(0x404040, 1.2);
    scene.add(ambientLight);

    // Hemisphere light for better ambient illumination
    const hemisphereLight = new THREE.HemisphereLight(0x8888ff, 0x444444, 0.6);
    scene.add(hemisphereLight);

    // Rim light (back light for silhouette)
    const rimLight1 = new THREE.DirectionalLight(0x6688ff, 0.8);
    rimLight1.position.set(-10, 5, -5);
    scene.add(rimLight1);

    const rimLight2 = new THREE.DirectionalLight(0x6688ff, 0.5);
    rimLight2.position.set(5, -5, -8);
    scene.add(rimLight2);

    // Planet (will be updated with 500 fault texture)
    const planetGeometry = new THREE.SphereGeometry(1, 64, 64);
    const planetMaterial = new THREE.MeshStandardMaterial({
      color: 0x808080,
      roughness: 0.8,
      metalness: 0.2,
    });
    const planet = new THREE.Mesh(planetGeometry, planetMaterial);
    scene.add(planet);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate planet
      planet.rotation.y += 0.001;
      
      // Slowly rotate stars
      stars.rotation.y += 0.0001;
      
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    sceneRef.current = { scene, camera, renderer, controls, planet, stars };

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      controls.dispose();
    };
  }, []);

  // Generate planet texture
  const generatePlanet = async () => {
    if (!pyodide || generating) return;

    setGenerating(true);
    try {
      // Generate 5 textures with different fault counts efficiently (cumulative)
      await pyodide.runPythonAsync(`
        import fractal_planet
        import numpy as np
        
        # Generate planet textures with fault counts: 10, 50, 100, 500
        # This generates them cumulatively for efficiency
        fault_counts = [10, 50, 100, 500]
        textures = fractal_planet.generate_multiple_planets(
            width=1024, 
            height=512, 
            fault_counts=fault_counts
        )
        print(f"Generated {len(textures)} textures (resolution: 1024x512)")
      `);

      const faultCounts = [10, 50, 100, 500];
      const newTextures: PlanetTexture[] = [];

      // Get each texture
      for (let i = 0; i < 4; i++) {
        const textureArray = pyodide.globals.get('textures').toJs()[i];
        const height = textureArray.length;
        const width = textureArray[0].length;
        
        // Convert to flat Uint8Array
        const data = new Uint8Array(height * width * 4);
        let idx = 0;
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const pixel = textureArray[y][x];
            data[idx++] = pixel[0];
            data[idx++] = pixel[1];
            data[idx++] = pixel[2];
            data[idx++] = pixel[3];
          }
        }
        
        // Create THREE.js texture
        const texture = new THREE.DataTexture(
          data,
          width,
          height,
          THREE.RGBAFormat,
          THREE.UnsignedByteType
        );
        texture.needsUpdate = true;
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

        // Create 2D canvas for map view
        const canvas2D = document.createElement('canvas');
        canvas2D.width = width;
        canvas2D.height = height;
        const ctx = canvas2D.getContext('2d')!;
        const imageData = ctx.createImageData(width, height);
        imageData.data.set(data);
        ctx.putImageData(imageData, 0, 0);

        newTextures.push({
          faultCount: faultCounts[i],
          texture,
          canvas2D,
        });

        // Update planet material with 500 fault texture (last one)
        if (i === 3 && sceneRef.current) {
          const { planet } = sceneRef.current;
          if (planet.material instanceof THREE.MeshStandardMaterial) {
            // Dispose old texture if exists
            if (planet.material.map) {
              planet.material.map.dispose();
            }
            
            // Create new material with the texture
            const newMaterial = new THREE.MeshStandardMaterial({
              map: texture,
              roughness: 0.8,
              metalness: 0.2,
            });
            
            // Dispose old material and assign new one
            planet.material.dispose();
            planet.material = newMaterial;
          }
        }
      }

      setTextures(newTextures);
      console.log('Planet and maps generated successfully!');
    } catch (error) {
      console.error('Error generating planet:', error);
      alert(`エラーが発生しました: ${error}`);
    } finally {
      setGenerating(false);
    }
  };

  // Auto-generate on mount
  useEffect(() => {
    if (pyodide && pyodideLoading && !generating) {
      generatePlanet();
    }
  }, [pyodide, pyodideLoading]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>🌍 Fractal Planet Generator</h1>
        <p className="subtitle">
          ランダムな断層線を使ってフラクタル地形を生成
        </p>
      </header>

      <div className="controls">
        <button
          onClick={generatePlanet}
          disabled={generating || !pyodideLoading || !!pyodideError}
          className="generate-button"
        >
          {generating ? '生成中...' : !pyodideLoading ? 'ロード中...' : '🔄 再生成'}
        </button>
        {pyodideError && (
          <div className="error-message">
            エラー: {pyodideError}
          </div>
        )}
      </div>

      <div className="main-content">
        {/* Left: Maps */}
        <div className="maps-section">
          <h2>2D Height Maps</h2>
          {textures.length === 0 ? (
            <div className="empty-state">
              <p>生成ボタンを押して地図を作成してください</p>
            </div>
          ) : (
            <div className="maps-list">
              {textures.map((item, index) => (
                <div 
                  key={index} 
                  className="map-item-compact"
                  onClick={() => setSelectedMap(item)}
                >
                  <div className="map-preview">
                    <img 
                      src={item.canvas2D.toDataURL()} 
                      alt={`Planet with ${item.faultCount} faults`}
                    />
                  </div>
                  <div className="map-label">
                    {item.faultCount} Faults
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: 3D View */}
        <div className="canvas-section">
          <h2>3D Planet (500 Faults)</h2>
          <div className="canvas-container">
            <canvas ref={canvasRef} />
            <div className="info-overlay">
              <p>🖱️ ドラッグ: 回転</p>
              <p>🖱️ スクロール: ズーム</p>
            </div>
          </div>
        </div>
      </div>

      {/* Modal for enlarged map */}
      {selectedMap && (
        <div className="modal-overlay" onClick={() => setSelectedMap(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedMap(null)}>
              ✕
            </button>
            <h3>{selectedMap.faultCount} Faults - Detailed View</h3>
            <div className="modal-image">
              <img 
                src={selectedMap.canvas2D.toDataURL()} 
                alt={`Planet with ${selectedMap.faultCount} faults`}
              />
            </div>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>
          このアプリは断層線アルゴリズムを使用してフラクタル地形を生成します。
          100個から500個までの断層線（大円）を累積的に配置し、各断層の両側で地形を上下させることで、
          リアルな惑星の地形を作り出します。解像度: 1024x512 | 3D惑星は500断層
        </p>
      </footer>
    </div>
  );
}

export default App;
