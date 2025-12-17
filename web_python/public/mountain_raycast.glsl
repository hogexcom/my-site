// Fractal Mountain Raycast
// 高さマップ: mountain_heightmap.png

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_time;
uniform sampler2D u_heightmap; // mountain_heightmap.png

#define MAX_STEPS 128
#define MAX_DIST 20.0
#define SURF_DIST 0.01
#define HEIGHT_SCALE 1.0

// カメラ設定
vec3 getCameraPosition(float time) {
    float angle = time * 0.1;
    float radius = 3.0;
    return vec3(
        cos(angle) * radius,
        2.5,
        sin(angle) * radius
    );
}

// 高さマップから高さを取得
float getHeight(vec2 pos) {
    // 0-1の範囲にラップ
    pos = fract(pos);
    float h = texture2D(u_heightmap, pos).r;
    return h * HEIGHT_SCALE;
}

// 地形のSDF (Signed Distance Function)
float terrainSDF(vec3 p) {
    vec2 xz = p.xz * 0.2 + 0.5; // スケール調整
    float h = getHeight(xz);
    return p.y - h;
}

// 法線計算
vec3 getNormal(vec3 p) {
    float d = terrainSDF(p);
    vec2 e = vec2(0.01, 0.0);
    vec3 n = d - vec3(
        terrainSDF(p - e.xyy),
        terrainSDF(p - e.yxy),
        terrainSDF(p - e.yyx)
    );
    return normalize(n);
}

// レイマーチング
float rayMarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        float d = terrainSDF(p);
        
        if (d < SURF_DIST) return t;
        if (t > MAX_DIST) break;
        
        t += d * 0.5; // 地形なので小さめのステップ
    }
    
    return -1.0;
}

// 空の色
vec3 getSkyColor(vec3 rd) {
    float t = rd.y * 0.5 + 0.5;
    vec3 skyTop = vec3(0.0824, 0.2196, 0.349);
    vec3 skyBottom = vec3(0.0941, 0.1176, 0.1412);
    return mix(skyBottom, skyTop, t);
}

// 山の色を高さで変える
vec3 getMountainColor(float height) {
    vec3 snow = vec3(0.8902, 0.9059, 0.7255);
    vec3 rock = vec3(0.0275, 0.0196, 0.0627);
    vec3 grass = vec3(0.0588, 0.0784, 0.1569);
    
    float t1 = smoothstep(0.3, 0.6, height / HEIGHT_SCALE);
    float t2 = smoothstep(0.7, 0.9, height / HEIGHT_SCALE);
    
    vec3 col = mix(grass, rock, t1);
    col = mix(col, snow, t2);
    
    return col;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    
    // カメラ設定
    vec3 ro = getCameraPosition(u_time);
    vec3 lookAt = vec3(0.0, 1.0, 0.0);
    vec3 forward = normalize(lookAt - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    
    // レイ方向
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);
    
    // レイマーチング
    float t = rayMarch(ro, rd);
    
    vec3 col;
    
    if (t > 0.0) {
        // 地形にヒット
        vec3 p = ro + rd * t;
        vec3 n = getNormal(p);
        
        // ライティング
        vec3 lightDir = normalize(vec3(0.5, 0.8, 0.3));
        float diff = max(dot(n, lightDir), 0.0);
        float ambient = 0.3;
        
        // 高さに応じた色
        vec3 baseCol = getMountainColor(p.y);
        
        // 最終色
        col = baseCol * (ambient + diff * 0.7);
        
        // フォグ
        float fogAmount = 1.0 - exp(-t * 0.02);
        col = mix(col, getSkyColor(rd), fogAmount);
    } else {
        // 空
        col = getSkyColor(rd);
    }
    
    // ガンマ補正
    col = pow(col, vec3(0.4545));
    
    gl_FragColor = vec4(col, 1.0);
}
