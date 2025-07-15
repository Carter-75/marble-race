'use client';

import React, { useRef, useEffect } from 'react';
import Matter from 'matter-js';

const MAX_METABALLS = 1000000; 

const vertexShaderSource = `
  attribute vec2 a_position;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const fragmentShaderSource = `
  precision highp float;
  #define MAX_METABALLS ${MAX_METABALLS}

  uniform sampler2D u_metaball_texture;
  uniform vec2 u_texture_dimensions;
  uniform int u_num_metaballs;
  uniform float u_time;
  uniform vec2 u_resolution;
  
  // 2D Simplex Noise by Stefan Gustavson
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
           + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m;
    m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 st = gl_FragCoord.xy;
    float totalInfluence = 0.0;
    vec3 mixedColor = vec3(0.0);

    for (int i = 0; i < MAX_METABALLS; i++) {
      if (i >= u_num_metaballs) break;

      float tex_x = mod(float(i), u_texture_dimensions.x);
      float tex_y = floor(float(i) / u_texture_dimensions.x);
      vec2 texCoord = (vec2(tex_x, tex_y) + 0.5) / u_texture_dimensions;
      vec4 metaball = texture2D(u_metaball_texture, texCoord).rgba;

      if (metaball.z == 0.0) continue;

      float dx = metaball.x - st.x;
      float dy = metaball.y - st.y;
      float r = metaball.z;
      float isWater = metaball.w;
      
      float influence = r * r / (dx * dx + dy * dy);
      totalInfluence += influence;

      vec3 color;
      if (isWater > 0.5) {
          // Blue/Teal
          vec3 lightBlue = vec3(0.4, 0.7, 1.0);
          vec3 teal = vec3(0.0, 0.5, 0.5);
          float noiseFactor = snoise(st * 0.005 + u_time * 0.0125);
          float colorFactor = (noiseFactor + 1.0) / 2.0;
          color = mix(lightBlue, teal, colorFactor);
      } else {
          // Red/Orange
          vec3 red = vec3(1.0, 0.1, 0.0);
          vec3 orange = vec3(1.0, 0.6, 0.0);
          float noiseFactor = snoise(st * 0.005 + u_time * 0.0125);
          float colorFactor = (noiseFactor + 1.0) / 2.0;
          color = mix(red, orange, colorFactor);
      }
      mixedColor += color * influence;
    }

    float threshold = 1.0;
    
    vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);

    if (totalInfluence > threshold) {
        mixedColor /= totalInfluence;
        finalColor = vec4(mixedColor, 1.0);
    }
    
    gl_FragColor = finalColor;
  }
`;

const LavaLamp: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl');
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }
    
    const floatTextureExt = gl.getExtension('OES_texture_float');
    if (!floatTextureExt) {
      console.error("Floating point textures not supported");
      return;
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    if (!vertexShader || !fragmentShader) return;

    const program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) return;

    const uniforms = {
      positionAttributeLocation: gl.getAttribLocation(program, "a_position"),
      resolutionUniformLocation: gl.getUniformLocation(program, "u_resolution"),
      metaballsTextureLocation: gl.getUniformLocation(program, "u_metaball_texture"),
      textureDimensionsUniformLocation: gl.getUniformLocation(program, "u_texture_dimensions"),
      numMetaballsUniformLocation: gl.getUniformLocation(program, "u_num_metaballs"),
      timeUniformLocation: gl.getUniformLocation(program, "u_time"),
    };

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // --- Data Texture for Metaballs ---
    const metaballTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, metaballTexture);
    const textureWidth = 1000;
    const textureHeight = 1000;
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, textureWidth, textureHeight, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // --- Matter.js setup ---
    const { Engine, World, Bodies, Composite } = Matter;
    const engine = Engine.create();
    engine.world.gravity.y = 1;

    const wallHeight = window.innerHeight * 100;
    const leftWall = Bodies.rectangle(-50, wallHeight / 2, 100, wallHeight, { isStatic: true });
    const rightWall = Bodies.rectangle(window.innerWidth + 50, wallHeight / 2, 100, wallHeight, { isStatic: true });
    
    // Add all bodies to the world
    World.add(engine.world, [leftWall, rightWall]);

    const ballInterval = setInterval(() => {
      const isWater = Math.random() > 0.5;
      const ball = Bodies.circle(
        Math.random() * window.innerWidth, -500,
        6 + Math.random() * 4,
        { 
          restitution: 0.5, 
          friction: 0.1,
          label: isWater ? 'water' : 'lava'
        }
      );
      World.add(engine.world, ball);
    }, 2);


    const resize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    };
    window.addEventListener('resize', resize);
    resize();
    
    let animationFrameId: number;
    const render = (time: number) => {
      Engine.update(engine);

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      gl.enableVertexAttribArray(uniforms.positionAttributeLocation);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.vertexAttribPointer(uniforms.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(uniforms.resolutionUniformLocation, gl.canvas.width, gl.canvas.height);
      gl.uniform1f(uniforms.timeUniformLocation, time * 0.001);
      
      const allBodies = Composite.allBodies(engine.world);
      const circleBodies = allBodies.filter(body => body.circleRadius != null);
      const numMetaballs = Math.min(circleBodies.length, MAX_METABALLS);

      gl.uniform1i(uniforms.numMetaballsUniformLocation, numMetaballs);

      if (numMetaballs > 0) {
          const data = new Float32Array(textureWidth * textureHeight * 4);
          for(let i=0; i<numMetaballs; i++) {
              const ball = circleBodies[i];
              const isWater = ball.label === 'water' ? 1.0 : 0.0;
              data[i * 4 + 0] = ball.position.x;
              data[i * 4 + 1] = gl.canvas.height - ball.position.y;
              data[i * 4 + 2] = ball.circleRadius || 0;
              data[i * 4 + 3] = isWater;
          }
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, metaballTexture);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, gl.RGBA, gl.FLOAT, data);
          gl.uniform1i(uniforms.metaballsTextureLocation, 0); // texture unit 0
      }
      
      gl.uniform2f(uniforms.textureDimensionsUniformLocation, textureWidth, textureHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      animationFrameId = requestAnimationFrame(render);
    }
    
    render(0);

    return () => {
        window.removeEventListener('resize', resize);
        clearInterval(ballInterval);
        cancelAnimationFrame(animationFrameId);
        World.clear(engine.world, false);
        Engine.clear(engine);
    }

  }, []);

  return <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, zIndex: 0 }} />;
};

function createShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) {
    return shader;
  }

  console.error(gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
  return null;
}

function createProgram(gl: WebGLRenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  const success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) {
    return program;
  }

  console.error(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
  return null;
}

export default LavaLamp; 