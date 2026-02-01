'use client';

import { useEffect, useRef } from 'react';

// Minimal OGL types for TypeScript
interface GL extends WebGLRenderingContext {
  canvas: HTMLCanvasElement;
  drawingBufferWidth: number;
  drawingBufferHeight: number;
}

const Prism = ({
  height = 3.5,
  baseWidth = 5.5,
  animationType = 'rotate',
  glow = 1,
  offset = { x: 0, y: 0 },
  noise = 0.5,
  transparent = true,
  scale = 3.6,
  hueShift = 0,
  colorFrequency = 1,
  timeScale = 0.5,
}: {
  height?: number;
  baseWidth?: number;
  animationType?: 'rotate' | 'hover' | '3drotate';
  glow?: number;
  offset?: { x: number; y: number };
  noise?: number;
  transparent?: boolean;
  scale?: number;
  hueShift?: number;
  colorFrequency?: number;
  timeScale?: number;
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const H = Math.max(0.001, height);
    const BW = Math.max(0.001, baseWidth);
    const BASE_HALF = BW * 0.5;
    const GLOW = Math.max(0.0, glow);
    const NOISE = Math.max(0.0, noise);
    const offX = offset?.x ?? 0;
    const offY = offset?.y ?? 0;
    const SAT = transparent ? 1.5 : 1;
    const SCALE = Math.max(0.001, scale);
    const HUE = hueShift || 0;
    const CFREQ = Math.max(0.0, colorFrequency || 1);
    const TS = Math.max(0, timeScale || 1);

    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl', { alpha: transparent, antialias: false }) as GL;
    if (!gl) return;

    Object.assign(canvas.style, {
      position: 'absolute',
      inset: '0',
      width: '100%',
      height: '100%',
      display: 'block',
    });
    container.appendChild(canvas);

    const vertex = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    const fragment = `
      precision highp float;
      uniform vec2 iResolution;
      uniform float iTime;
      uniform float uHeight;
      uniform float uBaseHalf;
      uniform mat3 uRot;
      uniform int uUseBaseWobble;
      uniform float uGlow;
      uniform vec2 uOffsetPx;
      uniform float uNoise;
      uniform float uSaturation;
      uniform float uScale;
      uniform float uHueShift;
      uniform float uColorFreq;
      uniform float uCenterShift;
      uniform float uInvBaseHalf;
      uniform float uInvHeight;
      uniform float uMinAxis;
      uniform float uPxScale;
      uniform float uTimeScale;

      vec4 tanh4(vec4 x) {
        vec4 e2x = exp(2.0 * x);
        return (e2x - 1.0) / (e2x + 1.0);
      }

      float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453123);
      }

      float sdOctaAnisoInv(vec3 p) {
        vec3 q = vec3(abs(p.x) * uInvBaseHalf, abs(p.y) * uInvHeight, abs(p.z) * uInvBaseHalf);
        float m = q.x + q.y + q.z - 1.0;
        return m * uMinAxis * 0.5773502691896258;
      }

      float sdPyramidUpInv(vec3 p) {
        float oct = sdOctaAnisoInv(p);
        float halfSpace = -p.y;
        return max(oct, halfSpace);
      }

      mat3 hueRotation(float a) {
        float c = cos(a), s = sin(a);
        mat3 W = mat3(0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114);
        mat3 U = mat3(0.701, -0.587, -0.114, -0.299, 0.413, -0.114, -0.300, -0.588, 0.886);
        mat3 V = mat3(0.168, -0.331, 0.500, 0.328, 0.035, -0.500, -0.497, 0.296, 0.201);
        return W + U * c + V * s;
      }

      void main() {
        vec2 f = (gl_FragCoord.xy - 0.5 * iResolution.xy - uOffsetPx) * uPxScale;
        float z = 5.0;
        float d = 0.0;
        vec3 p;
        vec4 o = vec4(0.0);
        float centerShift = uCenterShift;
        float cf = uColorFreq;

        mat2 wob = mat2(1.0);
        if (uUseBaseWobble == 1) {
          float t = iTime * uTimeScale;
          float c0 = cos(t + 0.0);
          float c1 = cos(t + 33.0);
          float c2 = cos(t + 11.0);
          wob = mat2(c0, c1, c2, c0);
        }

        const int STEPS = 100;
        for (int i = 0; i < STEPS; i++) {
          p = vec3(f, z);
          p.xz = p.xz * wob;
          p = uRot * p;
          vec3 q = p;
          q.y += centerShift;
          d = 0.1 + 0.2 * abs(sdPyramidUpInv(q));
          z -= d;
          o += (sin((p.y + z) * cf + vec4(0.0, 1.0, 2.0, 3.0)) + 1.0) / d;
        }

        o = tanh4(o * o * (uGlow) / 1e5);
        vec3 col = o.rgb;
        float n = rand(gl_FragCoord.xy + vec2(iTime));
        col += (n - 0.5) * uNoise;
        col = clamp(col, 0.0, 1.0);

        float L = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = clamp(mix(vec3(L), col, uSaturation), 0.0, 1.0);

        if (abs(uHueShift) > 0.0001) {
          col = clamp(hueRotation(uHueShift) * col, 0.0, 1.0);
        }

        gl_FragColor = vec4(col, o.a);
      }
    `;

    const createShader = (type: number, source: string) => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    };

    const vShader = createShader(gl.VERTEX_SHADER, vertex);
    const fShader = createShader(gl.FRAGMENT_SHADER, fragment);
    const program = gl.createProgram()!;
    gl.attachShader(program, vShader);
    gl.attachShader(program, fShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const uniforms: Record<string, WebGLUniformLocation | null> = {};
    [
      'iResolution',
      'iTime',
      'uHeight',
      'uBaseHalf',
      'uRot',
      'uUseBaseWobble',
      'uGlow',
      'uOffsetPx',
      'uNoise',
      'uSaturation',
      'uScale',
      'uHueShift',
      'uColorFreq',
      'uCenterShift',
      'uInvBaseHalf',
      'uInvHeight',
      'uMinAxis',
      'uPxScale',
      'uTimeScale',
    ].forEach((name) => {
      uniforms[name] = gl.getUniformLocation(program, name);
    });

    const resize = () => {
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
      gl.uniform2f(uniforms.iResolution, w, h);
      gl.uniform2f(uniforms.uOffsetPx, offX, offY);
      gl.uniform1f(uniforms.uPxScale, 1 / (h * 0.1 * SCALE));
    };

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();

    gl.uniform1f(uniforms.uHeight, H);
    gl.uniform1f(uniforms.uBaseHalf, BASE_HALF);
    gl.uniform1i(uniforms.uUseBaseWobble, animationType === 'rotate' ? 1 : 0);
    gl.uniform1f(uniforms.uGlow, GLOW);
    gl.uniform1f(uniforms.uNoise, NOISE);
    gl.uniform1f(uniforms.uSaturation, SAT);
    gl.uniform1f(uniforms.uScale, SCALE);
    gl.uniform1f(uniforms.uHueShift, HUE);
    gl.uniform1f(uniforms.uColorFreq, CFREQ);
    gl.uniform1f(uniforms.uCenterShift, H * 0.25);
    gl.uniform1f(uniforms.uInvBaseHalf, 1 / BASE_HALF);
    gl.uniform1f(uniforms.uInvHeight, 1 / H);
    gl.uniform1f(uniforms.uMinAxis, Math.min(BASE_HALF, H));
    gl.uniform1f(uniforms.uTimeScale, TS);

    const rotBuf = new Float32Array(9);
    rotBuf[0] = 1;
    rotBuf[4] = 1;
    rotBuf[8] = 1;
    gl.uniformMatrix3fv(uniforms.uRot, false, rotBuf);

    let raf = 0;
    const t0 = performance.now();

    const render = (t: number) => {
      const time = (t - t0) * 0.001;
      gl.uniform1f(uniforms.iTime, time);

      if (animationType === 'rotate') {
        const tScaled = time * TS;
        const yaw = tScaled * 0.3;
        const pitch = Math.sin(tScaled * 0.4) * 0.6;
        const roll = Math.sin(tScaled * 0.2) * 0.5;
        const cy = Math.cos(yaw), sy = Math.sin(yaw);
        const cx = Math.cos(pitch), sx = Math.sin(pitch);
        const cz = Math.cos(roll), sz = Math.sin(roll);
        rotBuf[0] = cy * cz + sy * sx * sz;
        rotBuf[1] = cx * sz;
        rotBuf[2] = -sy * cz + cy * sx * sz;
        rotBuf[3] = -cy * sz + sy * sx * cz;
        rotBuf[4] = cx * cz;
        rotBuf[5] = sy * sz + cy * sx * cz;
        rotBuf[6] = sy * cx;
        rotBuf[7] = -sx;
        rotBuf[8] = cy * cx;
        gl.uniformMatrix3fv(uniforms.uRot, false, rotBuf);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(render);
    };

    raf = requestAnimationFrame(render);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
      if (canvas.parentElement === container) container.removeChild(canvas);
    };
  }, [height, baseWidth, animationType, glow, noise, offset?.x, offset?.y, scale, transparent, hueShift, colorFrequency, timeScale]);

  return <div className="prism-container" ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%' }} />;
};

export default Prism;
