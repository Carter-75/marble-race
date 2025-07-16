'use client';
import React from 'react';
import styles from './page.module.css';
import LavaLamp from '../components/LavaLamp';

export default function Home() {
  return (
    <main className={styles.main}>
      <LavaLamp />
    </main>
  );
}
