'use client';
import React, { useRef, useEffect } from 'react';
import anime from 'animejs';
import styles from './page.module.css';
import LavaLamp from '../components/LavaLamp';

export default function Home() {
  const textRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const textElement = textRef.current;
    if (!textElement) return;

    // --- Anime.js pulsing text animation ---
    anime({
      targets: textElement,
      scale: [{ value: 1.05, duration: 1200 }, { value: 1, duration: 1200 }],
      loop: true,
      easing: 'easeInOutSine',
      direction: 'alternate',
    });
  }, []);

  return (
    <main className={styles.main}>
      <LavaLamp />
      <h1 ref={textRef} className={styles.title}>Marble Race</h1>
    </main>
  );
}
