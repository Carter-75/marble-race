'use client';

import React, { useRef, useEffect, useState } from 'react';
import Matter from 'matter-js';
import 'pathseg';
import anime from 'animejs';
const decomp = require('poly-decomp');

function debounce<F extends (...args: any[]) => any>(func: F, waitFor: number) {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  const debounced = (...args: Parameters<F>) => {
    if (timeout !== null) {
      clearTimeout(timeout);
      timeout = null;
    }
    timeout = setTimeout(() => func(...args), waitFor);
  };

  return debounced;
}


const MAX_METABALLS = 10000; 

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

type CameraMode = 'disabled' | 'winner' | 'mass' | 'loser';

const MARBLE_CATEGORY = 0x0001;
const WORLD_CATEGORY = 0x0002;
const SPINNER_CATEGORY = 0x0004;
const DYNAMIC_OBSTACLE_CATEGORY = 0x0008;

const LavaLamp: React.FC = () => {
  const [dimensions, setDimensions] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  });
  const engineRef = useRef<Matter.Engine | null>(null);
  const renderRef = useRef<Matter.Render | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const glProgramDataRef = useRef<{
      program: WebGLProgram;
      uniforms: { [key: string]: WebGLUniformLocation | null };
      positionBuffer: WebGLBuffer | null;
      metaballTexture: WebGLTexture | null;
      textureWidth: number;
      textureHeight: number;
  } | null>(null);
  const worldObjectsRef = useRef<{
      worldComposite: Matter.Composite,
      conveyorBelts: { body: Matter.Body, speed: number }[],
      oscillatingPegs: { body: Matter.Body, initialPos: Matter.Vector, amplitude: Matter.Vector, frequency: number }[],
      animatedCradles: { firstBob: Matter.Body, lastActivation: number, interval: number }[],
      allWindZones: { body: Matter.Body, indicator: Matter.Body, force: Matter.Vector }[],
      existingObstacles: Matter.Body[],
      finishAreaStartY: number,
      finalPlatformY: number,
  } | null>(null);


  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    
    const debouncedHandleResize = debounce(() => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }, 200);

    window.addEventListener('resize', debouncedHandleResize);
    return () => {
      window.removeEventListener('resize', debouncedHandleResize);
    };
  }, []);

  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const speedMultiplierRef = useRef(speedMultiplier);
  speedMultiplierRef.current = speedMultiplier;

  const [cameraMode, setCameraMode] = useState<CameraMode>('disabled');
  const cameraModeRef = useRef(cameraMode);
  cameraModeRef.current = cameraMode;

  const [isStarted, setIsStarted] = useState(false);
  const isStartedRef = useRef(isStarted);
  isStartedRef.current = isStarted;

  const [cameraTargetY, setCameraTargetY] = useState<number | null>(null);
  const cameraTargetYRef = useRef(cameraTargetY);
  cameraTargetYRef.current = cameraTargetY;

  const marbleTrails = useRef(new Map<number, Matter.Vector[]>());
  const stuckMarblesTracker = useRef(new Map<number, { position: Matter.Vector; timestamp: number }>());
  const metaballCanvasRef = useRef<HTMLCanvasElement>(null);
  const matterCanvasRef = useRef<HTMLCanvasElement>(null);
  const viewY = useRef(0);
  const marblesOnConveyors = useRef(new Map<number, Matter.Body[]>());


  useEffect(() => {
    const metaballCanvas = metaballCanvasRef.current;
    const matterCanvas = matterCanvasRef.current;
    if (!metaballCanvas || !matterCanvas) return;

    let animationFrameId: number;
    const { Engine, Render, World, Bodies, Composite, Constraint, Body, Common, Query, Sleeping } = Matter;
    type WindZoneController = { body: Matter.Body; indicator: Matter.Body; force: Matter.Vector; };

    // --- ONE-TIME INITIALIZATION ---
    if (!engineRef.current) {
        // --- Matter.js Engine and Renderer ---
        Common.setDecomp(decomp);
        const engine = Engine.create();
        engine.world.gravity.y = 1;
        engine.enableSleeping = true;
        engine.positionIterations = 10;
        engine.velocityIterations = 8;
        engine.constraintIterations = 5;
        (engine as any).ccdEnabled = true;
        engineRef.current = engine;
        
        const render = Render.create({
            canvas: matterCanvas,
            engine: engine,
            options: {
                width: dimensions.width,
                height: dimensions.height,
                wireframes: false,
                background: 'transparent',
            },
        });
        renderRef.current = render;

        // --- WebGL Context and Program ---
        const gl = metaballCanvas.getContext('webgl');
        if (gl) {
            glRef.current = gl;
    const floatTextureExt = gl.getExtension('OES_texture_float');
            if (floatTextureExt) {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
                if (vertexShader && fragmentShader) {
    const program = createProgram(gl, vertexShader, fragmentShader);
                    if (program) {
                        const textureWidth = 1000;
                        const textureHeight = 1000;
                        const metaballTexture = gl.createTexture();
                        gl.bindTexture(gl.TEXTURE_2D, metaballTexture);
                        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, textureWidth, textureHeight, 0, gl.RGBA, gl.FLOAT, null);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

                        glProgramDataRef.current = {
                            program,
                            uniforms: {
      positionAttributeLocation: gl.getAttribLocation(program, "a_position"),
      resolutionUniformLocation: gl.getUniformLocation(program, "u_resolution"),
      metaballsTextureLocation: gl.getUniformLocation(program, "u_metaball_texture"),
      textureDimensionsUniformLocation: gl.getUniformLocation(program, "u_texture_dimensions"),
      numMetaballsUniformLocation: gl.getUniformLocation(program, "u_num_metaballs"),
      timeUniformLocation: gl.getUniformLocation(program, "u_time"),
                            },
                            positionBuffer: gl.createBuffer(),
                            metaballTexture: metaballTexture,
                            textureWidth,
                            textureHeight,
    };
                        const positionBuffer = glProgramDataRef.current.positionBuffer;
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
                    }
                }
            }
        }
        
        // --- World Generation ---
        

        const courseLengthMultiplier = 3;
        const zoom = 1.5;
        const viewWidth = dimensions.width * zoom;
        const wallXOffset = (viewWidth - dimensions.width) / 2;
        
        const ramps = [];
        const rampWidth = 200;
        const numRampSets = 10 * courseLengthMultiplier;
        let currentCourseY = 500;

        for (let i = 0; i < numRampSets; i++) {
            const currentRampHeight = 400 + Math.random() * 800;
            const rampY = currentCourseY + currentRampHeight / 2;
            const isLeft = Math.random() > 0.5;
            let ramp;
            if (isLeft) {
                const x = -wallXOffset + rampWidth / 3;
                ramp = Bodies.fromVertices(x, rampY, [[{ x: 0, y: -currentRampHeight / 2 }, { x: rampWidth, y: 0 }, { x: 0, y: currentRampHeight / 2 }]], {
                    isStatic: true, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY }
                });
            } else {
                const x = dimensions.width + wallXOffset - rampWidth / 3;
                ramp = Bodies.fromVertices(x, rampY, [[{ x: rampWidth, y: -currentRampHeight / 2 }, { x: 0, y: 0 }, { x: rampWidth, y: currentRampHeight / 2 }]], {
                    isStatic: true, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY }
                });
            }
            ramps.push(ramp);
            const gapHeight = 200 + Math.random() * 600;
            const gapStartY = rampY + currentRampHeight / 2;
            const spikeHeightForSet = 30 + Math.random() * 50;
            for (let spikeY = gapStartY; spikeY < gapStartY + gapHeight; spikeY += spikeHeightForSet) {
                const spikeLength = 100 + Math.random() * 150;
                const spikeOptions = { isStatic: true, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } };
                let spike;
                if (!isLeft) {
                    const x = -wallXOffset + 10;
                    spike = Bodies.fromVertices(x, spikeY + spikeHeightForSet / 2, [[{ x: -spikeLength, y: -spikeHeightForSet / 2 }, { x: spikeLength, y: 0 }, { x: -spikeLength, y: spikeHeightForSet / 2 }]], spikeOptions);
                } else {
                    const x = dimensions.width + wallXOffset - 10;
                    spike = Bodies.fromVertices(x, spikeY + spikeHeightForSet / 2, [[{ x: spikeLength, y: -spikeHeightForSet / 2 }, { x: -spikeLength, y: 0 }, { x: spikeLength, y: spikeHeightForSet / 2 }]], spikeOptions);
                }
                ramps.push(spike);
            }
            currentCourseY = gapStartY + gapHeight;
        }

        const finishAreaStartY = currentCourseY + 100;
        const finishPlatforms: Matter.Body[] = [];
        const finishOptions = { isStatic: true, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } };
        let y = finishAreaStartY + 100;
        const verticalSpacing = 250;
        const numPlatforms = 6;
        const platformWidth = viewWidth * 0.9;
        for (let i = 0; i < numPlatforms; i++) {
            const angle = 0.15;
            let x;
            let effectiveAngle;
            if (i % 2 === 0) {
                x = -wallXOffset + platformWidth / 2 - 50;
                effectiveAngle = angle;
            } else {
                x = dimensions.width + wallXOffset - platformWidth / 2 + 50;
                effectiveAngle = -angle;
            }
            finishPlatforms.push(Bodies.rectangle(x, y, platformWidth, 20, { ...finishOptions, angle: effectiveAngle }));
            y += verticalSpacing;
        }

        const secondToLastPlatform = finishPlatforms[finishPlatforms.length - 1];
        const vertices = secondToLastPlatform.vertices;
        const leftMostVertex = [...vertices].sort((a, b) => a.x - b.x)[0];
        const finishWallHeight = 60;
        const finishWallWidth = 20;
        const wall = Bodies.rectangle(leftMostVertex.x + (finishWallWidth / 2), leftMostVertex.y - (finishWallHeight / 2) + 10, finishWallWidth, finishWallHeight, finishOptions);
        finishPlatforms.push(wall);

        const finalPlatformY = y + 100;
        const finalPlatform = Bodies.rectangle(dimensions.width / 2, finalPlatformY, viewWidth, 20, finishOptions);
        finishPlatforms.push(finalPlatform);

        const wallHeight = finalPlatformY + 100;
        const wallOptions = { isStatic: true, render: { visible: false }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } };
        const leftWall = Bodies.rectangle(-wallXOffset - 50, wallHeight / 2, 100, wallHeight, wallOptions);
        const rightWall = Bodies.rectangle(dimensions.width + wallXOffset + 50, wallHeight / 2, 100, wallHeight, wallOptions);
        const borderHeight = wallHeight;
        const borderOptions = { isStatic: true, render: { fillStyle: 'red' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } };
        const leftBorder = Bodies.rectangle(-wallXOffset, borderHeight / 2, 2, borderHeight, borderOptions);
        const rightBorder = Bodies.rectangle(dimensions.width + wallXOffset, borderHeight / 2, 2, borderHeight, borderOptions);

        const rect = Bodies.rectangle(dimensions.width / 2, 200, 400, 10, {
            isStatic: false, density: 0.00001, friction: 0, frictionAir: 0,
            render: { fillStyle: 'white' },
            collisionFilter: { category: SPINNER_CATEGORY, mask: MARBLE_CATEGORY }
        });
        const rectConstraint = Constraint.create({ pointA: { x: dimensions.width / 2, y: 200 }, bodyB: rect, stiffness: 1, length: 0 });

        const layeredObstacles: (Matter.Body | Matter.Constraint)[] = [];
        const conveyorBelts: { body: Matter.Body, speed: number }[] = [];
        const oscillatingPegs: { body: Matter.Body, initialPos: Matter.Vector, amplitude: Matter.Vector, frequency: number }[] = [];
        const animatedCradles: { firstBob: Matter.Body, lastActivation: number, interval: number }[] = [];
        const allWindZones: WindZoneController[] = [];

        const topSectionEndY = 1000;
        const obstacleLayers = [
            { type: 'platforms', count: 25 * courseLengthMultiplier / 2 },
            { type: 'spinners', count: 20 * courseLengthMultiplier / 2 },
            { type: 'funnels', count: 8 * courseLengthMultiplier / 2 },
            { type: 'plinko', count: 6 * courseLengthMultiplier / 2 },
            { type: 'cradles', count: 8 * courseLengthMultiplier / 2 },
            { type: 'flippers', count: 10 * courseLengthMultiplier / 2 },
            { type: 'conveyors', count: 12 * courseLengthMultiplier / 2 }
        ];

        const totalTrackHeight = finishAreaStartY - topSectionEndY - 200;
        const layerHeight = totalTrackHeight / obstacleLayers.length;
        let currentY = topSectionEndY;
        const existingObstacles: Matter.Body[] = [...ramps, ...finishPlatforms, rect];

        obstacleLayers.forEach(layer => {
            for (let i = 0; i < layer.count; i++) {
                let bodiesToAdd: (Matter.Body | Matter.Constraint)[] = [];
                let collisionCheckBody: Matter.Body | null = null;
                let isColliding = true;
                let attempts = 0;
                const maxAttempts = 50;

                do {
                    const x = -wallXOffset + 200 + Math.random() * (viewWidth - 400);
                    const y = currentY + Math.random() * layerHeight;
                    
                    switch (layer.type) {
                         case 'platforms': {
                            const width = 80 + Math.random() * 420;
                            const trackCenter = dimensions.width / 2;
                            const baseAngle = Math.random() * 0.4;
                            const angle = x < trackCenter ? baseAngle : -baseAngle;
                            const platform = Bodies.rectangle(x, y, width, 10, { isStatic: true, angle: angle, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } });
                            collisionCheckBody = platform;
                            bodiesToAdd = [platform];
                            break;
                        }
                         case 'spinners': {
                            const width = 70 + Math.random() * 230;
                            const spinner = Bodies.rectangle(x, y, width, 8, { density: 0.00001, friction: 0, frictionAir: 0, render: { fillStyle: 'white' }, collisionFilter: { category: SPINNER_CATEGORY, mask: MARBLE_CATEGORY } });
                            const pin = Constraint.create({ pointA: { x, y }, bodyB: spinner, stiffness: 1, length: 0 });
                            collisionCheckBody = spinner;
                            bodiesToAdd = [spinner, pin];
                            break;
                        }
                        case 'funnels': {
                            const funnelWidth = 150 + Math.random() * 250;
                            const funnelHeight = 120 + Math.random() * 180;
                            const wallThickness = 10;
                            const funnelWallOptions = { isStatic: true, render: { fillStyle: 'white' } };
                            const horizontalOffset = funnelWidth / 2.5;
                            const leftFunnelWall = Bodies.rectangle(-horizontalOffset, 0, wallThickness, funnelHeight, { ...funnelWallOptions, angle: Math.PI / 6 });
                            const rightFunnelWall = Bodies.rectangle(horizontalOffset, 0, wallThickness, funnelHeight, { ...funnelWallOptions, angle: -Math.PI / 6 });
                            const funnelBody = Body.create({ parts: [leftFunnelWall, rightFunnelWall], isStatic: true, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } });
                            Body.setPosition(funnelBody, { x, y });
                            collisionCheckBody = funnelBody;
                            bodiesToAdd = [funnelBody];
                            break;
                        }
                        case 'plinko': {
                            const rows = 5 + Math.floor(Math.random() * 5);
                            const cols = 7 + Math.floor(Math.random() * 6);
                            const pegRadius = 8;
                            const spacingX = 50;
                            const spacingY = 40;
                            const pegOptions = { isStatic: true, render: { fillStyle: 'white' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } };
                            const tempPegs: Matter.Body[] = [];
                            for (let row = 0; row < rows; row++) {
                                for (let col = 0; col < cols; col++) {
                                    const pegX = x - (cols * spacingX) / 2 + col * spacingX + (row % 2 === 1 ? spacingX / 2 : 0);
                                    const pegY = y - (rows * spacingY) / 2 + row * spacingY;
                                    const peg = Bodies.circle(pegX, pegY, pegRadius, pegOptions);
                                    oscillatingPegs.push({ body: peg, initialPos: { x: pegX, y: pegY }, amplitude: { x: 10 + Math.random() * 20, y: 5 + Math.random() * 10 }, frequency: 0.001 + Math.random() * 0.002 });
                                    tempPegs.push(peg);
                                }
                            }
                            if (tempPegs.length > 0) {
                                const bounds = Matter.Bounds.create(tempPegs.map(p => p.vertices));
                                collisionCheckBody = Bodies.rectangle((bounds.min.x + bounds.max.x) / 2, (bounds.min.y + bounds.max.y) / 2, bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, { isStatic: true });
                                bodiesToAdd = tempPegs;
                            }
                            break;
                        }
                        case 'cradles': {
                            const numBobs = 5;
                            const bobRadius = 15;
                            const ropeLength = 100;
                            const cradleBobs: (Matter.Body | Matter.Constraint)[] = [];
                            const tempParts: Matter.Body[] = [];
                            for (let j = 0; j < numBobs; j++) {
                                const bobX = x - (numBobs * bobRadius * 2) / 2 + bobRadius + j * bobRadius * 2;
                                const bobY = y + ropeLength;
                                const bob = Bodies.circle(bobX, bobY, bobRadius, { density: 0.1, restitution: 1, friction: 0, render: { fillStyle: 'white' }, collisionFilter: { category: DYNAMIC_OBSTACLE_CATEGORY, mask: MARBLE_CATEGORY | DYNAMIC_OBSTACLE_CATEGORY } });
                                const rope = Constraint.create({ pointA: { x: bobX, y: y }, bodyB: bob, length: ropeLength, stiffness: 0.9 });
                                cradleBobs.push(bob, rope);
                                tempParts.push(bob);
                            }
                            if (tempParts.length > 0) {
                                const firstBob = cradleBobs.find(item => 'type' in item && item.type === 'body') as Matter.Body;
                                if (firstBob) {
                                    animatedCradles.push({ firstBob: firstBob, lastActivation: 0, interval: 3000 + Math.random() * 4000 });
                                }
                                const bounds = Matter.Bounds.create(tempParts.map(p => p.vertices));
                                collisionCheckBody = Bodies.rectangle((bounds.min.x + bounds.max.x) / 2, (bounds.min.y + bounds.max.y) / 2, bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, { isStatic: true });
                                bodiesToAdd = cradleBobs;
                            }
                            break;
                        }
                        case 'flippers': {
                            const flipperWidth = 100 + Math.random() * 80;
                            const flipperHeight = 15;
                            const flipperGap = 10;
                            const flipperOptions = { density: 0.1, render: { fillStyle: 'white' }, collisionFilter: { category: DYNAMIC_OBSTACLE_CATEGORY, mask: MARBLE_CATEGORY } };
                            const leftFlipper = Bodies.rectangle(x - flipperWidth / 2 - flipperGap, y, flipperWidth, flipperHeight, { ...flipperOptions, angle: -0.3 });
                            const rightFlipper = Bodies.rectangle(x + flipperWidth / 2 + flipperGap, y, flipperWidth, flipperHeight, { ...flipperOptions, angle: 0.3 });
                            const leftPivot = { x: leftFlipper.position.x - flipperWidth / 2, y: y };
                            const rightPivot = { x: rightFlipper.position.x + flipperWidth / 2, y: y };
                            const leftConstraint = Constraint.create({ pointA: leftPivot, bodyB: leftFlipper, stiffness: 0.1 });
                            const rightConstraint = Constraint.create({ pointA: rightPivot, bodyB: rightFlipper, stiffness: 0.1 });
                            const flipperBodies = [leftFlipper, rightFlipper];
                            const bounds = Matter.Bounds.create(flipperBodies.flatMap(f => f.vertices));
                            collisionCheckBody = Bodies.rectangle((bounds.min.x + bounds.max.x) / 2, (bounds.min.y + bounds.max.y) / 2, bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, { isStatic: true });
                            bodiesToAdd = [leftFlipper, rightFlipper, leftConstraint, rightConstraint];
                            break;
                        }
                        case 'conveyors': {
                            const beltWidth = 200 + Math.random() * 300;
                            const beltHeight = 15;
                            const beltSpeed = Math.random() > 0.5 ? 2 : -2;
                            const conveyorBody = Bodies.rectangle(x, y, beltWidth, beltHeight, { isStatic: true, render: { fillStyle: beltSpeed > 0 ? '#44DD44' : '#DD4444' }, collisionFilter: { category: WORLD_CATEGORY, mask: MARBLE_CATEGORY } });
                            conveyorBelts.push({ body: conveyorBody, speed: beltSpeed });
                            collisionCheckBody = conveyorBody;
                            bodiesToAdd = [conveyorBody];
                            break;
                        }
                    }
                    isColliding = collisionCheckBody ? Query.collides(collisionCheckBody, existingObstacles).length > 0 : true;
                    attempts++;
                } while (isColliding && attempts < maxAttempts);

                if (!isColliding && collisionCheckBody) {
                    layeredObstacles.push(...bodiesToAdd);
                    const bodiesOnly = bodiesToAdd.filter((item): item is Matter.Body => 'type' in item && item.type === 'body');
                    existingObstacles.push(...bodiesOnly);
                }
            }

            if (layer !== obstacleLayers[obstacleLayers.length - 1]) {
                const separatorY = currentY + layerHeight;
                const separator = Bodies.rectangle(dimensions.width / 2, separatorY, viewWidth, 5, {
                    isSensor: true, isStatic: true, render: { fillStyle: 'rgba(255, 255, 255, 0.2)' }
                });
                layeredObstacles.push(separator);
            }
            currentY += layerHeight;
        });

        const mainWindZoneHeight = 600;
        const mainWindZoneY = 200 + mainWindZoneHeight / 2;
        const mainWindZoneXOffset = 250;
        const initialMainWindZoneX = dimensions.width / 2 - mainWindZoneXOffset;
        const mainWindZoneBody = Bodies.rectangle(initialMainWindZoneX, mainWindZoneY, 200, mainWindZoneHeight, { isSensor: true, isStatic: true, render: { strokeStyle: 'rgba(128, 128, 128, 0.5)', fillStyle: 'transparent', lineWidth: 1 }});
        const mainIndicatorBody = Bodies.rectangle(initialMainWindZoneX, mainWindZoneY + mainWindZoneHeight / 2 + 25, 200, 50, { isStatic: true, render: { fillStyle: 'red' }, label: 'MainWindZoneIndicator' });
        allWindZones.push({ body: mainWindZoneBody, indicator: mainIndicatorBody, force: { x: 0, y: -0.0005 } });

        const worldComposite = Composite.create({ label: 'World' });
        Composite.add(worldComposite, [
            leftWall, rightWall, leftBorder, rightBorder, 
            rect, rectConstraint, ...ramps,
            ...layeredObstacles,
            mainWindZoneBody, mainIndicatorBody,
            ...finishPlatforms
        ]);
        
        worldObjectsRef.current = {
            worldComposite,
            conveyorBelts,
            oscillatingPegs,
            animatedCradles,
            allWindZones,
            existingObstacles,
            finishAreaStartY,
            finalPlatformY,
        };

        World.add(engine.world, worldObjectsRef.current.worldComposite);
    }

    const engine = engineRef.current;
    const render = renderRef.current;
    const gl = glRef.current;
    const glProgramData = glProgramDataRef.current;

    if (!engine || !render || !gl || !glProgramData || !worldObjectsRef.current) {
        return;
    }
    
    // --- Resize handler ---
    render.options.width = dimensions.width;
    render.options.height = dimensions.height;
    metaballCanvas.width = dimensions.width;
    metaballCanvas.height = dimensions.height;
    matterCanvas.width = dimensions.width;
    matterCanvas.height = dimensions.height;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    
    // --- Get generated data from refs ---
    if (!worldObjectsRef.current) return;
    const { conveyorBelts, oscillatingPegs, animatedCradles, allWindZones, existingObstacles, finishAreaStartY, finalPlatformY } = worldObjectsRef.current;
    
    let ballInterval: NodeJS.Timeout | null = null;
    let stopSpawningTimeout: NodeJS.Timeout | null = null;

    const handleCollisionStart = (event: Matter.IEventCollision<Matter.Engine>) => {
        event.pairs.forEach(pair => {
            const { bodyA, bodyB } = pair;
            const conveyor = conveyorBelts.find(c => c.body.id === bodyA.id || c.body.id === bodyB.id);
            const marble = (conveyor?.body.id === bodyA.id) ? bodyB : bodyA;

            if (conveyor && (marble.label === 'water' || marble.label === 'lava')) {
                const marbles = marblesOnConveyors.current.get(conveyor.body.id) || [];
                if (!marbles.includes(marble)) {
                    marbles.push(marble);
                    marblesOnConveyors.current.set(conveyor.body.id, marbles);
                }
            }
        });
    };

    const handleCollisionEnd = (event: Matter.IEventCollision<Matter.Engine>) => {
        event.pairs.forEach(pair => {
            const { bodyA, bodyB } = pair;
            const conveyor = conveyorBelts.find(c => c.body.id === bodyA.id || c.body.id === bodyB.id);
            const marble = (conveyor?.body.id === bodyA.id) ? bodyB : bodyA;

            if (conveyor && (marble.label === 'water' || marble.label === 'lava')) {
                const marbles = marblesOnConveyors.current.get(conveyor.body.id);
                if (marbles) {
                    const index = marbles.indexOf(marble);
                    if (index > -1) {
                        marbles.splice(index, 1);
                        marblesOnConveyors.current.set(conveyor.body.id, marbles);
                    }
                }
            }
        });
    };

    Matter.Events.on(engine, 'collisionStart', handleCollisionStart);
    Matter.Events.on(engine, 'collisionEnd', handleCollisionEnd);

    const startSpawning = () => {
        if (ballInterval) return;
        const zoom = 1.5;
        const viewWidth = dimensions.width * zoom;
        const wallXOffset = (viewWidth - dimensions.width) / 2;
        
        
        ballInterval = setInterval(() => {
      const isWater = Math.random() > 0.5;
            const spawnPadding = 10;
            const spawnX = -wallXOffset + spawnPadding + Math.random() * (viewWidth - spawnPadding * 2);
            const spawnY = viewY.current - (dimensions.height * zoom - dimensions.height) / 2 - 50;
            const initialRestitution = 0.85 + Math.random() * 0.1;
            const ball = Bodies.circle(spawnX, spawnY, 6 + Math.random() * 4, {
                restitution: initialRestitution,
                friction: 0,
                frictionAir: 0.02 + Math.random() * 0.04,
                density: 0.0008 + Math.random() * 0.0004,
                label: isWater ? 'water' : 'lava',
                collisionFilter: {
                    category: MARBLE_CATEGORY,
                    mask: WORLD_CATEGORY | SPINNER_CATEGORY | DYNAMIC_OBSTACLE_CATEGORY | MARBLE_CATEGORY
                },
                render: { visible: false }
            });
            (ball as any).initialRestitution = initialRestitution;
            marbleTrails.current.set(ball.id, []);
            World.add(engine.world, ball);
        }, 2);
        stopSpawningTimeout = setTimeout(() => {
            if (ballInterval) clearInterval(ballInterval);
        }, 10000);
    };

    const handleWheel = (e: WheelEvent) => {
      setCameraMode('disabled');
      setCameraTargetY(null);
      viewY.current += e.deltaY * 0.5; // Adjust scroll sensitivity
    };
    window.addEventListener('wheel', handleWheel);

    const findDensestCluster = (bodies: Matter.Body[]): Matter.Body[] => {
        if (bodies.length < 5) {
            return bodies;
        }

        const sortedBodies = [...bodies].sort((a, b) => a.position.y - b.position.y);
        const windowSize = Math.max(10, Math.floor(sortedBodies.length * 0.2));

        if (sortedBodies.length <= windowSize) {
            return sortedBodies;
        }

        let minSpread = Infinity;
        let bestWindowStartIndex = 0;

        for (let i = 0; i <= sortedBodies.length - windowSize; i++) {
            const window = sortedBodies.slice(i, i + windowSize);
            const spread = window[window.length - 1].position.y - window[0].position.y;
            if (spread < minSpread) {
                minSpread = spread;
                bestWindowStartIndex = i;
            }
        }

        return sortedBodies.slice(bestWindowStartIndex, bestWindowStartIndex + windowSize);
    }

    // Initial render of the static world
    Render.world(render);

    let lastTime = 0;
    const renderLoop = (time: number) => {
      const isRunning = isStartedRef.current;
      const speed = isRunning ? speedMultiplierRef.current : 1.0;
      
      const zoom = 1.5;
      const viewWidth = dimensions.width * zoom;
      const viewHeight = dimensions.height * zoom;
      const offsetX = (viewWidth - dimensions.width) / 2;
      const offsetY = (viewHeight - dimensions.height) / 2;

      if (isRunning) {
        lastTime = time;
        startSpawning();

        // --- Get all marbles once for efficiency ---
        const allMarbles = Composite.allBodies(engine.world).filter(body => body.label === 'water' || body.label === 'lava');

        // --- Dynamically adjust restitution for settling ---
        allMarbles.forEach(marble => {
          const speed = Matter.Vector.magnitude(marble.velocity);
          const isOnFinalPlatform = marble.position.y > finalPlatformY - 20;
          const initialRestitution = (marble as any).initialRestitution || 0.9;

          if (isOnFinalPlatform && speed < 1) {
            if (marble.restitution !== 0) {
              Body.set(marble, 'restitution', 0);
            }
          } else {
            if (marble.restitution !== initialRestitution) {
              Body.set(marble, 'restitution', initialRestitution);
            }
          }
        });
        
        // --- Teleport Stuck Marbles ---
        const stuckTimeThreshold = 3000; // 3 seconds
        const stuckDistanceThreshold = 5; // pixels
        const teleportDistance = 50; // pixels
        allMarbles.forEach(marble => {
          // Don't teleport marbles that have finished
          if (marble.position.y > finishAreaStartY) {
            return;
          }

          const now = time;
          const tracker = stuckMarblesTracker.current;
          const lastState = tracker.get(marble.id);

          if (!lastState) {
              tracker.set(marble.id, { position: Matter.Vector.clone(marble.position), timestamp: now });
              return;
          }

          const distanceMoved = Matter.Vector.magnitude(
              Matter.Vector.sub(marble.position, lastState.position)
          );

          if (distanceMoved < stuckDistanceThreshold) {
              if (now - lastState.timestamp > stuckTimeThreshold) {
                  Body.setPosition(marble, {
                      x: marble.position.x,
                      y: marble.position.y + teleportDistance,
                  });
                  Body.setVelocity(marble, { x: 0, y: 0 });
                  tracker.set(marble.id, { position: Matter.Vector.clone(marble.position), timestamp: now });
              }
          } else {
              tracker.set(marble.id, { position: Matter.Vector.clone(marble.position), timestamp: now });
          }
        });

        // --- Camera Controls ---
        if (cameraModeRef.current !== 'disabled' && allMarbles.length > 0) {
          let targetY;
          switch (cameraModeRef.current) {
            case 'winner':
              targetY = Math.max(...allMarbles.map(b => b.position.y));
              break;
            case 'loser':
              targetY = Math.min(...allMarbles.map(b => b.position.y));
              break;
            case 'mass': {
              const cluster = findDensestCluster(allMarbles);
              if (cluster.length > 0) {
                  const totalY = cluster.reduce((sum, b) => sum + b.position.y, 0);
                  targetY = totalY / cluster.length;
              }
              break;
            }
          }

          if (targetY !== undefined) {
            setCameraTargetY(targetY);
          }
        }

        if (cameraTargetYRef.current !== null) {
            const targetViewY = cameraTargetYRef.current - dimensions.height / 2;
            const smoothing = Math.min(1.0, 0.1 * Math.sqrt(speedMultiplierRef.current));
            viewY.current += (targetViewY - viewY.current) * smoothing;
        }

        // --- Apply All Wind Forces ---
        const circleBodies = Composite.allBodies(engine.world).filter(body => body.circleRadius != null);
        allWindZones.forEach(zone => {
          const bodiesInZone = Query.collides(zone.body, circleBodies);
          bodiesInZone.forEach((collision: Matter.Collision) => {
            const body = collision.bodyA.isSensor ? collision.bodyB : collision.bodyA;
             if (body && !body.isStatic) {
              Body.applyForce(body, body.position, Matter.Vector.mult(zone.force, speed));
            }
          });
        });

        // --- Apply Conveyor Belt Force ---
        conveyorBelts.forEach(conveyor => {
          const marblesOnThisConveyor = marblesOnConveyors.current.get(conveyor.body.id) || [];
          
          marblesOnThisConveyor.forEach(marble => {
              const force = { x: conveyor.speed * 0.001 * speed, y: 0 };
              Body.applyForce(marble, marble.position, force);
          });
        });

        // --- Enforce Speed Limit ---
        const softMaxSpeed = 15;
        const hardMaxSpeed = 30;
        allMarbles.forEach(body => {
          const speed = Matter.Vector.magnitude(body.velocity);
          if (speed > hardMaxSpeed) {
            // Hard cap to prevent extreme speeds
            Body.setVelocity(body, Matter.Vector.mult(body.velocity, hardMaxSpeed / speed));
          } else if (speed > softMaxSpeed) {
            // Gently slow down particles that are moving too fast with progressive damping
            const damping = 1 - ( (speed - softMaxSpeed) / (hardMaxSpeed - softMaxSpeed) * 0.05 );
            Body.setVelocity(body, Matter.Vector.mult(body.velocity, damping));
          }
        });
      }
      
      // --- ALWAYS-ON ANIMATIONS ---
      const currentTime = isRunning ? time : lastTime;
      const centerX = dimensions.width / 2;
      const moveRange = 250; 
      const newX = centerX + Math.cos(currentTime * 0.0002 * speed) * moveRange;
      const mainWindZone = allWindZones.find(z => z.indicator.label === 'MainWindZoneIndicator');
       if(mainWindZone) {
          Body.setPosition(mainWindZone.body, { x: newX, y: mainWindZone.body.position.y });
          Body.setPosition(mainWindZone.indicator, { x: newX, y: mainWindZone.indicator.position.y });
       }
      
      const randomWindZonesOnly = allWindZones.filter(z => z.indicator.label !== 'MainWindZoneIndicator');
      randomWindZonesOnly.forEach(zone => {
        const moveSpeed = 0.5 * speed;
        const velocity = Matter.Vector.mult(Matter.Vector.normalise(zone.force), moveSpeed);
        const nextPos = Matter.Vector.add(zone.body.position, velocity);
        
        // Create a temporary body to check for future collisions
        const tempBody = Bodies.rectangle(nextPos.x, nextPos.y, zone.body.vertices[1].x - zone.body.vertices[0].x, zone.body.vertices[2].y - zone.body.vertices[1].y, { angle: zone.body.angle });
        
        if (Query.collides(tempBody, existingObstacles).length === 0) {
            Body.setPosition(zone.body, nextPos);
            const nextIndicatorPos = Matter.Vector.add(zone.indicator.position, velocity);
            Body.setPosition(zone.indicator, nextIndicatorPos);
        }
      });

      const engineTime = engine.timing.timestamp;
      animatedCradles.forEach(cradle => {
          if (engineTime - cradle.lastActivation > cradle.interval) {
              const cradleSpeed = Matter.Vector.magnitude(cradle.firstBob.velocity);
              if (cradleSpeed < 0.1) {
                  Body.applyForce(cradle.firstBob, cradle.firstBob.position, { x: -0.05 * speed, y: -0.01 * speed });
                  cradle.lastActivation = engineTime;
              }
          }
      });

      oscillatingPegs.forEach(peg => {
          const pegTime = engine.timing.timestamp;
          const newX = peg.initialPos.x + Math.sin(pegTime * peg.frequency) * peg.amplitude.x;
          const newY = peg.initialPos.y + Math.cos(pegTime * peg.frequency) * peg.amplitude.y;
          Body.setPosition(peg.body, { x: newX, y: newY });
      });

      // --- Physics Culling (Sleeping) ---
      const buffer = dimensions.height; 
      const activeZoneTop = (viewY.current - offsetY) - buffer;
      const activeZoneBottom = (viewY.current + dimensions.height + offsetY) + buffer;

      const allBodies = Composite.allBodies(engine.world);

      const gridCellSize = 200;
      const spatialGrid = new Map<string, Matter.Body[]>();

      for (const body of allBodies) {
        if (body.isStatic) continue;

        const isInActiveZone = body.position.y >= activeZoneTop && body.position.y <= activeZoneBottom;
        if (isInActiveZone) {
          Sleeping.set(body, false);

          const startX = Math.floor(body.bounds.min.x / gridCellSize);
          const startY = Math.floor(body.bounds.min.y / gridCellSize);
          const endX = Math.floor(body.bounds.max.x / gridCellSize);
          const endY = Math.floor(body.bounds.max.y / gridCellSize);

          for (let i = startX; i <= endX; i++) {
              for (let j = startY; j <= endY; j++) {
                  const key = `${i},${j}`;
                  if (!spatialGrid.has(key)) {
                      spatialGrid.set(key, []);
                  }
                  spatialGrid.get(key)!.push(body);
              }
          }

        }
      }

      for (const cell of Array.from(spatialGrid.values())) {
          for (let i = 0; i < cell.length; i++) {
              for (let j = i + 1; j < cell.length; j++) {
                  const bodyA = cell[i];
                  const bodyB = cell[j];
                  if (bodyA.isStatic || bodyB.isStatic) continue;

                  // Custom broadphase check
                  if (Matter.Bounds.overlaps(bodyA.bounds, bodyB.bounds)) {
                      const collision = Matter.Collision.collides(bodyA, bodyB, undefined);
                      if (collision && collision.collided) {
                         // This is a simplified stand-in for a proper narrowphase + resolution
                         // In a real scenario, you'd dispatch collision events here.
                      }
                  }
              }
          }
      }

      for (const body of Composite.allBodies(engine.world)) {
        if (body.isStatic) continue;

        const isInActiveZone = body.position.y >= activeZoneTop && body.position.y <= activeZoneBottom;
        if (isInActiveZone) {
          Sleeping.set(body, false);
        }
      }
      
      // --- Physics Update Loop ---
      const baseDelta = 1000 / 60;
      const subSteps = Math.ceil(speed);
      const subDelta = baseDelta * speed / subSteps;

      for (let i = 0; i < subSteps; i++) {
        Engine.update(engine, subDelta);
        
        if (isRunning) {
          // --- Update marble trails for each substep ---
          const allMarbles = Composite.allBodies(engine.world).filter(body => body.label === 'water' || body.label === 'lava');
          allMarbles.forEach(marble => {
              const trail = marbleTrails.current.get(marble.id);
              if (trail) {
                  trail.push(Matter.Vector.clone(marble.position));
                  const speed = Matter.Vector.magnitude(marble.velocity);
                  const maxTrailLength = Math.min(40, Math.floor(speed * 2.5));
                  while (trail.length > maxTrailLength) {
                      trail.shift();
                  }
              }
          });
        }
      }

      // --- RENDERING (always runs) ---
      Render.lookAt(render, {
        min: { x: -offsetX, y: viewY.current - offsetY },
        max: { x: dimensions.width + offsetX, y: viewY.current + dimensions.height + offsetY }
      });
      
      const ctx = matterCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
      
      Render.world(render);

      // --- Draw Trails & Arrows ---
      const viewMinX = -offsetX;
      const viewMinY = viewY.current - offsetY;
      
      if (ctx && isRunning) {
        const allMarbles = Composite.allBodies(engine.world).filter(body => body.label === 'water' || body.label === 'lava');
        
        // --- Draw Trails ---
        ctx.lineWidth = 3;
        allMarbles.forEach(marble => {
          const trail = marbleTrails.current.get(marble.id);
          if (!trail || trail.length < 2) return;

          const isWater = marble.label === 'water';
          const baseColor = isWater ? '64, 178, 255' : '255, 100, 0';

          for (let i = 0; i < trail.length - 1; i++) {
              const startPointWorld = trail[i];
              const endPointWorld = trail[i+1];

              const relStartX = (startPointWorld.x - viewMinX) / viewWidth;
              const relStartY = (startPointWorld.y - viewMinY) / viewHeight;
              const startCanvasX = relStartX * ctx.canvas.width;
              const startCanvasY = relStartY * ctx.canvas.height;

              const relEndX = (endPointWorld.x - viewMinX) / viewWidth;
              const relEndY = (endPointWorld.y - viewMinY) / viewHeight;
              const endCanvasX = relEndX * ctx.canvas.width;
              const endCanvasY = relEndY * ctx.canvas.height;
              
              const alpha = (i / trail.length) * 0.7;

              ctx.beginPath();
              ctx.moveTo(startCanvasX, startCanvasY);
              ctx.lineTo(endCanvasX, endCanvasY);
              ctx.strokeStyle = `rgba(${baseColor}, ${alpha})`;
              ctx.stroke();
          }
        });
      }
      
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        
        allWindZones.forEach(zone => {
          const pos = zone.indicator.position; // Use the indicator's center for the arrow
          const relX = (pos.x - viewMinX) / viewWidth;
          const relY = (pos.y - viewMinY) / viewHeight;
          const canvasX = relX * ctx.canvas.width;
          const canvasY = relY * ctx.canvas.height;
          
          ctx.save();
          ctx.translate(canvasX, canvasY);
          ctx.rotate(zone.indicator.angle);
          ctx.beginPath();
          ctx.moveTo(0, -8);
          ctx.lineTo(5, 2);
          ctx.lineTo(-5, 2);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        });
      }


      const glContext = glRef.current;
      if (!glContext) {
        animationFrameId = requestAnimationFrame(renderLoop);
        return;
      }
      
      glContext.clearColor(0, 0, 0, 0);
      glContext.clear(glContext.COLOR_BUFFER_BIT);

      glContext.useProgram(glProgramData.program);

      glContext.enableVertexAttribArray(glProgramData.uniforms.positionAttributeLocation as number);
      glContext.bindBuffer(glContext.ARRAY_BUFFER, glProgramData.positionBuffer);
      glContext.vertexAttribPointer(glProgramData.uniforms.positionAttributeLocation as number, 2, glContext.FLOAT, false, 0, 0);

      glContext.uniform2f(glProgramData.uniforms.resolutionUniformLocation, glContext.canvas.width, glContext.canvas.height);
      glContext.uniform1f(glProgramData.uniforms.timeUniformLocation, (isRunning ? time : lastTime) * 0.001);
      
      const allBodiesForRender = Composite.allBodies(engine.world);
      const circleBodiesForRender = allBodiesForRender.filter(body => body.circleRadius != null);
      const numMetaballs = Math.min(circleBodiesForRender.length, MAX_METABALLS);

      glContext.uniform1i(glProgramData.uniforms.numMetaballsUniformLocation, numMetaballs);

      if (numMetaballs > 0) {
          const data = new Float32Array(glProgramData.textureWidth * glProgramData.textureHeight * 4);
          const viewMinX = -offsetX;
          const viewMinY = viewY.current - offsetY;

          for(let i=0; i<numMetaballs; i++) {
              const ball = circleBodiesForRender[i];
              const isWater = ball.label === 'water' ? 1.0 : 0.0;

              // Transform world coords to zoomed/panned screen coords
              const relX = (ball.position.x - viewMinX) / viewWidth;
              const relY = (ball.position.y - viewMinY) / viewHeight;
              
              data[i * 4 + 0] = relX * glContext.canvas.width;
              data[i * 4 + 1] = glContext.canvas.height - (relY * glContext.canvas.height);
              data[i * 4 + 2] = (ball.circleRadius || 0) / zoom;
              data[i * 4 + 3] = isWater;
          }
          glContext.activeTexture(glContext.TEXTURE0);
          glContext.bindTexture(glContext.TEXTURE_2D, glProgramData.metaballTexture);
          glContext.texSubImage2D(glContext.TEXTURE_2D, 0, 0, 0, glProgramData.textureWidth, glProgramData.textureHeight, glContext.RGBA, glContext.FLOAT, data);
          glContext.uniform1i(glProgramData.uniforms.metaballsTextureLocation, 0);
      }
      
      glContext.uniform2f(glProgramData.uniforms.textureDimensionsUniformLocation, glProgramData.textureWidth, glProgramData.textureHeight);
      glContext.drawArrays(glContext.TRIANGLES, 0, 6);

      animationFrameId = requestAnimationFrame(renderLoop);
    }
    
    renderLoop(0);

    return () => {
      window.removeEventListener('wheel', handleWheel);
      Matter.Events.off(engine, 'collisionStart', handleCollisionStart);
      Matter.Events.off(engine, 'collisionEnd', handleCollisionEnd);
        if (ballInterval) clearInterval(ballInterval);
      if (stopSpawningTimeout) clearTimeout(stopSpawningTimeout);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      
      if (render) {
        Render.stop(render);
      }
      // Don't clear engineRef and other refs here, as they should persist
        marbleTrails.current.clear();
        marblesOnConveyors.current.clear();
        stuckMarblesTracker.current.clear();
    };
  }, [dimensions]);

  const getCameraModeText = () => {
    switch (cameraMode) {
      case 'disabled': return 'Auto Follow: Off';
      case 'winner': return 'Following: Winner';
      case 'mass': return 'Following: Center of Mass';
      case 'loser': return 'Following: Loser';
      default: return 'Auto Follow';
    }
  };

  const handleStartClick = () => {
    setIsStarted(true);
    anime({
      targets: '.title',
      opacity: 0,
      duration: 1500,
      easing: 'easeInOutQuad',
      delay: 500,
    });
  };

  const handleModeChange = () => {
    setCameraMode(prevMode => {
        const newMode = (() => {
          if (prevMode === 'disabled') return 'winner';
          if (prevMode === 'winner') return 'mass';
          if (prevMode === 'mass') return 'loser';
          if (prevMode === 'loser') return 'disabled';
          return 'disabled';
        })();

        if (newMode === 'disabled') {
            setCameraTargetY(null);
        }
        
        return newMode;
    });
  };

  return (
    <>
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 20, color: 'white', fontFamily: 'sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '15px' }}>
        {!isStarted && (
            <button
              onClick={handleStartClick}
              style={{ padding: '8px 12px', cursor: 'pointer', background: 'rgba(0,0,0,0.5)', border: '1px solid white', color: 'white', borderRadius: '5px', width: '220px', textAlign: 'center' }}
            >
              Start
            </button>
        )}
        <button
          onClick={handleModeChange}
          style={{ padding: '8px 12px', cursor: 'pointer', background: 'rgba(0,0,0,0.5)', border: '1px solid white', color: 'white', borderRadius: '5px', width: '220px', textAlign: 'center' }}
        >
          {getCameraModeText()}
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(0,0,0,0.5)', padding: '8px', borderRadius: '5px' }}>
          <span>Speed: {speedMultiplier.toFixed(1)}x</span>
          <input
            type="range"
            min="1"
            max="10"
            step="0.1"
            value={speedMultiplier}
            onChange={(e) => setSpeedMultiplier(parseFloat(e.target.value))}
            style={{ cursor: 'pointer' }}
          />
        </div>
      </div>
      <h1 className="title" style={{
        position: 'absolute',
        top: '8%',
        left: '50%',
        transform: 'translateX(-50%)',
        color: 'white',
        fontSize: '4rem',
        zIndex: 10,
        pointerEvents: 'none'
      }}>Marble Race</h1>
      <canvas ref={metaballCanvasRef} style={{ position: 'absolute', top: 0, left: 0, zIndex: 0 }} />
      <canvas ref={matterCanvasRef} style={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }} />
    </>
  );
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
  console.log(gl.getShaderInfoLog(shader));
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
  console.log(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
  return null;
}

export default LavaLamp; 