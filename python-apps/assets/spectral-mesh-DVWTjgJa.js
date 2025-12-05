import"./modulepreload-polyfill-B5Qt9EMX.js";import{r as f,B as F,a as g,j as e,D as _,T as S,C as N,G as C,O as w,c as M}from"./vendor-three-CePzC98V.js";import{u as A}from"./usePyodide-SVK5obCv.js";import"./vendor-react-CCYiWZgt.js";function j({mesh:t,color:o,position:i,label:a}){const l=f.useMemo(()=>{const r=new F,s=new Float32Array(t.vertices.flat());r.setAttribute("position",new g(s,3));const n=new Uint32Array(t.faces.flat());return r.setIndex(new g(n,1)),r.computeVertexNormals(),r},[t]);return e.jsxs("group",{position:i,children:[e.jsx("mesh",{geometry:l,children:e.jsx("meshStandardMaterial",{color:o,side:_,flatShading:!1})}),e.jsx(S,{position:[0,-.8,0],fontSize:.1,color:"white",anchorX:"center",anchorY:"middle",children:a})]})}function E({onFileSelect:t,onFilter:o,cutoffRatio:i,onCutoffChange:a,numEigenpairs:l,onNumEigenpairsChange:r,isProcessing:s,isReady:n}){const p=c=>{const d=c.target.files?.[0];d&&t(d)};return e.jsxs("div",{className:"control-panel",children:[e.jsx("h2",{children:"Spectral Mesh Filter"}),e.jsxs("div",{className:"control-group",children:[e.jsx("label",{htmlFor:"file-upload",children:"Upload OBJ File"}),e.jsx("input",{id:"file-upload",type:"file",accept:".obj",onChange:p,disabled:s||!n})]}),e.jsxs("div",{className:"control-group",children:[e.jsxs("label",{htmlFor:"eigenpairs",children:["Eigenpairs: ",l]}),e.jsx("input",{id:"eigenpairs",type:"range",min:"10",max:"300",step:"10",value:l,onChange:c=>r(Number(c.target.value)),disabled:s}),e.jsxs("div",{className:"range-labels",children:[e.jsx("span",{children:"10"}),e.jsx("span",{children:"300"})]})]}),e.jsxs("div",{className:"control-group",children:[e.jsxs("label",{htmlFor:"cutoff",children:["Cutoff Ratio: ",i.toFixed(2)]}),e.jsx("input",{id:"cutoff",type:"range",min:"0.1",max:"0.9",step:"0.05",value:i,onChange:c=>a(Number(c.target.value)),disabled:s}),e.jsxs("div",{className:"range-labels",children:[e.jsx("span",{children:"0.1"}),e.jsx("span",{children:"0.9"})]})]}),e.jsx("button",{className:"filter-button",onClick:o,disabled:s||!n,children:s?"Processing...":"Apply Lowpass Filter"}),!n&&e.jsx("div",{className:"status",children:"Initializing Pyodide..."})]})}function O(){const[t,o]=f.useState(null),[i,a]=f.useState(null),[l,r]=f.useState(!1),[s,n]=f.useState(.3),[p,c]=f.useState(100),{pyodide:d,isReady:m,error:x}=A(["spectral_mesh_processing.py"]),y=async u=>{try{const h=await u.text(),b=B(h);o(b),a(null)}catch(h){console.error("Failed to load model:",h)}},v=async()=>{if(!(!t||!d||!m)){r(!0);try{await d.runPythonAsync(`
import numpy as np
from spectral_mesh_processing import SpectralMeshProcessor

# メッシュデータを設定
vertices = np.array(${JSON.stringify(t.vertices)})
faces = np.array(${JSON.stringify(t.faces)})

print(f"頂点数: {len(vertices)}, 面数: {len(faces)}")

# スペクトル処理
processor = SpectralMeshProcessor(vertices, faces, dual_type='circumcentric')

# スペクトル分解
print(f"スペクトル分解開始 (${p}固有対)...")
processor.compute_spectrum(num_eigenpairs=${p}, verbose=False)

# ローパスフィルター適用
cutoff_freq = processor.nyquist_frequency * ${s}
print(f"ローパスフィルター適用 (カットオフ: {cutoff_freq:.2f})...")
filtered_vertices = processor.apply_lowpass_filter(
    cutoff_freq=cutoff_freq,
    include_residual=False,
    verbose=False
)

# 結果を格納
result_vertices = filtered_vertices.tolist()
result_faces = faces.tolist()

print("処理完了")
      `);const u=d.globals.get("result_vertices").toJs(),h=d.globals.get("result_faces").toJs();a({vertices:Array.from(u),faces:Array.from(h)})}catch(u){console.error("Filter error:",u)}finally{r(!1)}}};return e.jsxs("div",{className:"app-container",children:[e.jsx(E,{onFileSelect:y,onFilter:v,cutoffRatio:s,onCutoffChange:n,numEigenpairs:p,onNumEigenpairsChange:c,isProcessing:l,isReady:m}),e.jsxs("div",{className:"canvas-container",children:[x&&e.jsxs("div",{style:{color:"red",padding:"1rem"},children:["Error: ",x]}),e.jsxs(N,{camera:{position:[2,2,2],fov:50},children:[e.jsx("color",{attach:"background",args:["#0a0a0a"]}),e.jsx("ambientLight",{intensity:.5}),e.jsx("directionalLight",{position:[10,10,5],intensity:1}),e.jsx("directionalLight",{position:[-10,-10,-5],intensity:.3}),t&&e.jsx(j,{mesh:t,color:"#3b82f6",position:[-.6,0,0],label:"Original"}),i&&e.jsx(j,{mesh:i,color:"#22c55e",position:[.6,0,0],label:"Filtered"}),e.jsx(C,{args:[10,10],cellColor:"#333",sectionColor:"#555"}),e.jsx(w,{enableDamping:!0,dampingFactor:.05,rotateSpeed:.5,zoomSpeed:.3,panSpeed:.5})]})]}),e.jsx("footer",{className:"app-footer",children:e.jsx("p",{children:e.jsx("a",{href:"/my-site/python-apps/",children:"← Back to Apps"})})})]})}function B(t){const o=[],i=[],a=t.split(`
`);for(const l of a){const r=l.trim();if(!r||r.startsWith("#"))continue;const s=r.split(/\s+/);if(s[0]==="v")o.push([parseFloat(s[1]),parseFloat(s[2]),parseFloat(s[3])]);else if(s[0]==="f"){const n=s.slice(1,4).map(p=>parseInt(p.split("/")[0])-1);n.length===3&&i.push(n)}}return{vertices:o,faces:i}}M.createRoot(document.getElementById("root")).render(e.jsx(f.StrictMode,{children:e.jsx(O,{})}));
