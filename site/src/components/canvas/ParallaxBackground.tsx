'use client';

import { useEffect, useRef, useCallback } from 'react';
import { setCameraListener } from '@/engine/juice';

const RATES = [0.05, 0.12, 0.35, 0.65];
const REST_X = 0;
const REST_Y = 30;

// --- Alignment constants ---
// Game camera defaults (from constants.ts STAGE.camera)
const CAM_TOP = 120;
const CAM_BOTTOM = -60;
// Ground (y=0) screen fraction from top of camera view at rest
const GROUND_FRAC = CAM_TOP / (CAM_TOP - CAM_BOTTOM); // 0.6667

// SVG platform surface center ≈ y=93 in viewBox 0 0 250 173
const SVG_PLAT_Y = 93;
const SVG_W = 250;
const SVG_H = 173;
const SVG_ASPECT = SVG_W / SVG_H;

// Layer 1: Edge terrain + nebula formations (SVG lines 4-12, 34-42)
const Layer1 = () => (
  <svg viewBox="0 0 250 173" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" fill="none" style={{ width: '100%', height: '100%' }}>
    <path d="M 2.46,1.87 L 45.17,1.41 L 46.12,7.41 L 50.04,10.11 L 59.68,10.21 L 68.78,15.1 L 70.14,19.83 L 66.07,20.84 L 54.16,19.06 L 46.92,18.26 L 37.97,14.4 L 26.77,12.8 L 18.01,14.4 L 7.01,12.8 L 2.46,10.69 V 1.87 Z" fill="#1A2A3B"/>
    <path d="M 90.12,0.41 L 100.16,0.41 L 123.71,0.41 L 126.75,4.88 L 130.41,4.11 L 137.74,1.41 L 145.21,1.92 L 149.13,0.41 H 176.23 L 177.31,7.24 L 181.7,10.69 L 186.46,7.24 L 187.91,1.41 H 192.03 L 196.64,3.92 L 200.08,1.41 L 206.11,1.61 L 207.66,6.18 L 211.88,1.41 H 247.41 L 247.96,7.41 L 244.31,11.07 L 239.75,11.21 L 235.9,17.21 L 227.81,18.46 L 222.28,21.31 L 217.32,17.81 L 217.98,14.1 L 211.98,15.1 L 204.09,12.2 L 207.09,7.41 L 202.97,6.18 L 197.13,8.88 L 191.1,12.2 L 189.65,19.06 L 195.98,19.83 L 200.29,20.84 L 206.93,25.21 L 212.2,33.41 L 216.96,30.71 L 219.66,34.81 L 225.89,36.11 L 232.83,34.81 L 238.86,37.11 L 242.11,45.71 L 247.96,55.11 V 85.81 L 244.11,90.11 L 247.96,105.01 V 172.41 L 227.09,172.21 L 237.71,163.41 L 230.77,158.91 L 222.28,160.71 L 211.28,166.01 L 200.49,169.31 L 191.1,170.31 L 187.91,167.21 L 182.85,168.11 L 176.23,172.21 L 167.63,171.91 L 172.09,165.31 L 168.94,162.11 L 162.11,165.31 L 159.21,171.71 L 151.32,171.91 L 149.33,166.01 L 143.19,168.11 L 139.75,171.91 L 129.15,171.71 L 126.75,166.01 L 121.69,171.91 L 112.4,171.71 L 109.11,166.01 L 103.07,169.31 L 99.22,171.91 L 89.03,171.71 L 87.38,166.01 L 79.89,168.11 L 74.83,171.91 L 66.23,171.71 L 63.74,166.01 L 56.4,171.91 L 47.91,171.71 L 38.42,168.11 L 26.77,170.31 L 2.46,170.01 V 141.01 L 6.01,131.01 L 2.46,124.61 V 102.21 L 5.81,96.01 L 2.46,85.81 V 70.31 L 9.8,63.11 L 17.89,58.21 L 15.79,54.31 L 2.46,55.11 V 36.11 L 10.65,31.61 L 12.95,28.41 L 2.46,29.31 V 17.21 L 12.85,17.81 L 23.45,17.21 L 30.78,15.1 L 22.09,13.7 L 12.95,12.2 L 6.01,10.69 L 2.46,7.41 V 1.87 L 2.46,1.87 Z" fill="#2C1A48" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 247.96,105.81 L 242.91,109.21 L 239.06,107.01 L 236.11,111.71 L 238.71,115.21 L 232.18,118.01 L 231.43,122.71 L 236.11,123.21 L 231.43,127.11 L 236.11,128.81 L 241.95,125.91 L 247.96,124.61 V 105.81 Z" fill="#354063" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 247.96,130.11 L 242.51,132.91 L 238.11,136.01 L 234.66,134.61 L 231.12,136.81 L 220.32,139.11 L 225.77,144.01 L 230.72,144.81 L 236.11,147.01 L 241.16,145.61 L 247.96,139.11 V 130.11 Z" fill="#514584" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 247.96,156.91 L 242.81,161.61 L 237.71,160.71 L 239.75,166.01 L 234.21,171.91 L 247.96,171.71 V 156.91 Z" fill="#472D68" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 240.81,46.01 L 237.91,37.71 L 235.01,36.11 L 235.91,41.01 L 233.22,43.01 L 237.71,47.11 L 239.61,54.31 L 243.11,51.91 L 240.81,46.01 Z" fill="#542A22" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 124.21,0.41 L 126.75,4.88 L 130.41,4.11 L 132.71,7.41 L 131.31,10.69 L 125.61,9.08 L 116.42,4.11 L 117.12,0.41 H 124.21 Z" fill="#0D182D" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 176.23,1.41 L 177.31,7.24 L 181.7,10.69 L 186.46,7.24 L 187.91,1.41 H 176.23 Z" fill="#161228" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 45.17,1.41 L 46.12,7.41 L 50.04,10.11 L 53.38,8.88 L 50.04,6.18 L 48.09,4.11 L 47.02,1.41 H 45.17 Z" fill="#2B2424" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    {/* Lines 34-42: edge terrain continued */}
    <path d="M 2.46,34.81 L 11.55,34.11 L 13.65,31.61 L 18.84,31.11 L 17.89,28.41 L 12.95,28.41 L 2.46,32.21 V 34.81 Z" fill="#102A39" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 2.46,46.01 L 9.8,44.31 L 13.65,42.31 L 17.89,41.01 L 20.79,38.41 L 26.77,37.11 L 20.79,34.81 L 14.75,36.11 L 9.8,39.71 L 2.46,41.01 V 46.01 Z" fill="#193551" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 2.46,69.81 L 6.01,65.91 L 11.55,62.41 L 17.89,60.11 L 10.65,67.61 L 4.71,74.81 L 2.46,76.11 V 69.81 Z" fill="#3D1A56" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 2.46,109.21 L 8.4,113.91 L 17.89,116.61 L 12.2,119.31 L 20.79,122.71 L 15.1,124.61 L 27.92,132.91 L 2.46,128.81 V 109.21 Z" fill="#311E4E" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 180.25,34.81 L 184.61,36.11 L 187.11,38.41 L 184.61,37.11 L 180.25,34.81 Z" fill="#193551" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 194.98,25.21 L 199.29,26.11 L 203.94,28.41 L 206.93,33.41 L 202.19,32.21 L 197.13,32.21 L 194.98,25.21 Z" fill="#165973" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 182.15,43.01 L 188.18,45.71 L 192.03,51.91 L 197.78,53.31 L 196.64,57.41 L 190.3,55.11 L 180.25,46.01 L 182.15,43.01 Z" fill="#6A2490" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 239.21,63.81 L 244.75,67.61 L 247.96,75.61 L 243.91,80.81 L 239.21,76.11 L 241.16,70.31 L 239.21,63.81 Z" fill="#301D56" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 2.46,139.11 L 11.55,139.91 L 19.74,144.81 L 26.77,147.01 L 32.02,153.41 L 26.77,154.91 L 36.12,165.31 L 2.46,164.01 V 139.11 Z" fill="#3A1C58" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
  </svg>
);

// Star twinkle timing presets — cycled by index for organic stagger
const STAR_DUR = [2.5, 3.1, 2.8, 3.7, 4.3, 2.9, 3.4, 3.9];
const STAR_DEL = [0, 1.2, 0.4, 2.1, 0.8, 1.7, 0.3, 2.6];
const ss = (i: number) => ({ animationDuration: `${STAR_DUR[i % 8]}s`, animationDelay: `${STAR_DEL[i % 8]}s` });

// Layer 2: Stars + small highlights (SVG lines 13-33, 43-70)
const Layer2 = () => (
  <svg viewBox="0 0 250 173" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" fill="none" style={{ width: '100%', height: '100%' }}>
    <path className="star" style={ss(0)} d="M 47.12,8.01 L 46.12,9.61 L 47.91,10.11 L 48.51,8.51 L 47.12,8.01 Z" fill="#FFA766" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 115.91,12.2 L 115.31,13.7 L 116.71,13.9 L 116.91,12.4 L 115.91,12.2 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 37.07,15.1 L 36.72,16.11 L 37.77,16.31 L 37.97,15.2 L 37.07,15.1 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 18.84,1.41 L 18.49,2.41 L 19.54,2.61 L 19.74,1.51 L 18.84,1.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 212.28,0.91 L 211.93,1.91 L 212.98,2.11 L 213.18,1.01 L 212.28,0.91 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(5)} d="M 98.32,1.41 L 97.97,2.41 L 99.02,2.61 L 99.22,1.51 L 98.32,1.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 66.23,1.41 L 65.88,2.41 L 66.93,2.61 L 67.13,1.51 L 66.23,1.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(7)} d="M 136.89,3.11 L 136.54,4.11 L 137.59,4.31 L 137.79,3.21 L 136.89,3.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(0)} d="M 230.02,11.07 L 229.67,12.07 L 230.72,12.27 L 230.92,11.17 L 230.02,11.07 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 235.91,26.11 L 235.56,27.11 L 236.61,27.31 L 236.81,26.21 L 235.91,26.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 231.42,33.41 L 230.72,34.81 L 232.12,35.11 L 232.42,33.71 L 231.42,33.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 222.93,45.01 L 222.23,46.41 L 223.62,46.71 L 223.92,45.31 L 222.93,45.01 Z" fill="#FFD79D" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 212.88,11.7 L 212.53,12.7 L 213.58,12.9 L 213.78,11.8 L 212.88,11.7 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(5)} d="M 103.07,15.1 L 102.72,16.11 L 103.77,16.31 L 103.97,15.2 L 103.07,15.1 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 131.31,22.01 L 130.96,23.01 L 132.01,23.21 L 132.21,22.11 L 131.31,22.01 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(7)} d="M 132.01,33.41 L 131.31,34.81 L 132.71,35.11 L 133.01,33.71 L 132.01,33.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(0)} d="M 118.31,34.81 L 117.96,35.81 L 119.01,36.01 L 119.21,34.91 L 118.31,34.81 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 228.62,20.84 L 228.27,21.84 L 229.32,22.04 L 229.52,20.94 L 228.62,20.84 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 189.65,17.21 L 189.3,18.21 L 190.35,18.41 L 190.55,17.31 L 189.65,17.21 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 183.61,25.21 L 183.26,26.21 L 184.31,26.41 L 184.51,25.31 L 183.61,25.21 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 50.04,17.21 L 49.69,18.21 L 50.74,18.41 L 50.94,17.31 L 50.04,17.21 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    {/* Lines 43-70: bottom stars + misc highlights */}
    <path className="star" style={ss(5)} d="M 46.12,170.31 L 49.11,170.71 L 49.71,169.31 L 48.21,168.71 L 46.12,170.31 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 224.32,154.11 L 227.32,154.51 L 227.92,153.11 L 226.42,152.51 L 224.32,154.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(7)} d="M 224.32,167.21 L 225.42,167.61 L 226.02,166.61 L 224.92,166.01 L 224.32,167.21 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(0)} d="M 232.12,154.11 L 232.82,154.51 L 233.32,153.71 L 232.52,153.31 L 232.12,154.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 200.29,157.61 L 200.99,158.01 L 201.49,157.21 L 200.69,156.81 L 200.29,157.61 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 200.49,167.61 L 201.19,168.01 L 201.69,167.21 L 200.89,166.81 L 200.49,167.61 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 210.78,169.31 L 211.28,169.61 L 211.68,168.91 L 211.08,168.61 L 210.78,169.31 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 226.02,129.71 L 226.72,130.11 L 227.22,129.31 L 226.42,128.91 L 226.02,129.71 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(5)} d="M 219.08,138.51 L 217.38,139.71 L 216.38,137.81 L 218.48,137.11 L 219.08,138.51 Z" fill="#96C9FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 208.63,145.61 L 204.79,146.21 L 201.49,148.11 L 196.64,146.21 L 198.64,143.41 L 204.09,143.41 L 208.63,145.61 Z" fill="#50B1D8" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path d="M 237.91,136.81 L 235.61,137.81 L 234.21,136.01 L 236.11,134.61 L 237.91,136.81 Z" fill="#8A74B4" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 19.09,163.41 L 18.49,164.21 L 19.49,164.71 L 19.99,163.71 L 19.09,163.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(7)} d="M 27.92,165.31 L 27.57,166.01 L 28.37,166.31 L 28.67,165.51 L 27.92,165.31 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(0)} d="M 12.95,143.41 L 12.6,144.11 L 13.4,144.41 L 13.7,143.61 L 12.95,143.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 13.15,150.71 L 12.55,151.91 L 13.85,152.31 L 14.35,150.91 L 13.15,150.71 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 45.17,150.01 L 44.57,151.21 L 45.87,151.61 L 46.37,150.21 L 45.17,150.01 Z" fill="#96C9FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 199.99,39.71 L 199.39,40.91 L 200.69,41.31 L 201.19,39.91 L 199.99,39.71 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 199.14,36.11 L 198.54,37.31 L 199.84,37.71 L 200.34,36.31 L 199.14,36.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(5)} d="M 223.62,49.11 L 223.02,50.31 L 224.32,50.71 L 224.82,49.31 L 223.62,49.11 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 231.42,73.11 L 230.02,75.61 L 232.82,76.61 L 233.81,73.81 L 231.42,73.11 Z" fill="#96C9FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(7)} d="M 8.4,77.51 L 7.21,79.41 L 9.2,80.41 L 10.3,78.21 L 8.4,77.51 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(0)} d="M 17.89,83.61 L 17.29,84.81 L 18.59,85.21 L 19.09,83.81 L 17.89,83.61 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(1)} d="M 29.33,79.41 L 28.73,80.61 L 30.03,81.01 L 30.53,79.61 L 29.33,79.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(2)} d="M 12.95,128.01 L 12.55,128.81 L 13.45,129.11 L 13.75,128.21 L 12.95,128.01 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(3)} d="M 20.79,61.51 L 20.39,62.31 L 21.29,62.61 L 21.59,61.71 L 20.79,61.51 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(4)} d="M 20.79,143.41 L 20.39,144.21 L 21.29,144.51 L 21.59,143.61 L 20.79,143.41 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(5)} d="M 230.72,141.61 L 230.32,142.41 L 231.22,142.71 L 231.52,141.81 L 230.72,141.61 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
    <path className="star" style={ss(6)} d="M 241.16,158.01 L 240.76,158.81 L 241.66,159.11 L 241.96,158.21 L 241.16,158.01 Z" fill="#F1E3FF" stroke="black" strokeWidth="0.2" strokeMiterlimit="10"/>
  </svg>
);

// Layer 3: Translucent towers (SVG lines 126-127 + defs 153-162)
const Layer3 = () => (
  <svg viewBox="0 0 250 173" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" fill="none" style={{ width: '100%', height: '100%' }}>
    <defs>
      <linearGradient id="parallax-paint6" x1="80.682" y1="15.0996" x2="80.682" y2="83.806" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F6C6FF"/>
        <stop offset="0.510417" stopColor="#824BD1"/>
        <stop offset="1" stopColor="#483699"/>
      </linearGradient>
      <linearGradient id="parallax-paint7" x1="159.54" y1="6.10645" x2="159.54" y2="82.6065" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F6C6FF"/>
        <stop offset="0.510417" stopColor="#824BD1"/>
        <stop offset="1" stopColor="#483699"/>
      </linearGradient>
    </defs>
    <path opacity="0.5" d="M 92.02,81.81 L 87.43,23.21 L 80.69,15.1 L 74.28,21.31 L 69.93,82.21 L 72.68,82.01 L 73.63,71.01 L 70.78,70.31 L 65.33,73.11 L 61.24,83.81 L 67.13,83.11 L 74.28,82.61 L 80.69,82.21 L 90.18,81.91 L 98.02,81.61 L 96.12,78.71 L 94.62,66.71 L 91.78,63.21 L 89.63,64.81 L 90.18,81.61" fill="url(#parallax-paint6)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path opacity="0.5" d="M 142.84,81.11 L 147.99,56.01 L 150.12,54.31 L 152.22,57.41 L 154.42,14.21 L 163.21,6.11 L 168.39,12.2 L 169.39,71.01 L 170.74,58.91 L 173.51,58.21 L 176.23,60.81 L 177.73,77.01 L 179.75,79.41 V 82.61 L 174.86,81.61 L 166.69,80.81 L 153.72,80.11 L 143.94,80.41" fill="url(#parallax-paint7)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
  </svg>
);

// Layer 4: Platform complex + supports (SVG lines 71-125 + defs 128-152)
const Layer4 = () => (
  <svg viewBox="0 0 250 173" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" fill="none" style={{ width: '100%', height: '100%' }}>
    <defs>
      <linearGradient id="parallax-paint0" x1="65.8127" y1="137.806" x2="65.8127" y2="172.41" gradientUnits="userSpaceOnUse">
        <stop stopColor="#494199"/>
        <stop offset="1" stopColor="#A978D5"/>
      </linearGradient>
      <linearGradient id="parallax-paint1" x1="94.5507" y1="140.11" x2="94.5507" y2="172.41" gradientUnits="userSpaceOnUse">
        <stop stopColor="#392A72"/>
        <stop offset="1" stopColor="#9A50CC"/>
      </linearGradient>
      <linearGradient id="parallax-paint2" x1="124.676" y1="136.706" x2="124.676" y2="172.41" gradientUnits="userSpaceOnUse">
        <stop stopColor="#392A72"/>
        <stop offset="1" stopColor="#A978D5"/>
      </linearGradient>
      <linearGradient id="parallax-paint3" x1="169.169" y1="138.506" x2="169.169" y2="172.41" gradientUnits="userSpaceOnUse">
        <stop stopColor="#494199"/>
        <stop offset="1" stopColor="#A978D5"/>
      </linearGradient>
      <linearGradient id="parallax-paint4" x1="184.131" y1="137.806" x2="184.131" y2="169.31" gradientUnits="userSpaceOnUse">
        <stop stopColor="#494199"/>
        <stop offset="1" stopColor="#A978D5"/>
      </linearGradient>
      <linearGradient id="parallax-paint5" x1="38.2662" y1="127.106" x2="38.2662" y2="144.81" gradientUnits="userSpaceOnUse">
        <stop stopColor="#494199"/>
        <stop offset="1" stopColor="#A978D5"/>
      </linearGradient>
    </defs>
    {/* Main platform body */}
    <path d="M 2.46,102.21 C 2.46,92.11 33.07,81.11 125.21,81.11 C 217.34,81.11 242.51,90.11 237.71,102.21 V 106.51 C 237.71,110.81 234.81,113.21 227.87,116.11 L 215.15,129.71 L 187.11,151.21 V 166.01 L 184.61,170.31 L 179.75,168.11 L 176.23,172.21 H 162.11 L 157.81,168.11 L 151.32,171.91 H 136.89 L 132.71,170.31 L 126.75,171.91 L 120.06,170.31 L 112.4,171.91 H 103.07 L 97.92,168.11 L 89.03,171.71 L 79.89,170.31 L 74.83,171.91 H 60.7 L 56.4,168.11 L 47.91,171.71 L 36.12,170.31 L 2.46,170.01 V 102.21 Z" fill="#391A51" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 11.75,96.41 C 11.75,88.91 38.92,81.11 125.21,81.11 C 211.51,81.11 237.91,88.91 237.91,96.41 C 237.91,103.91 211.51,111.71 125.21,111.71 C 38.92,111.71 11.75,103.91 11.75,96.41 Z" fill="#9B36B5" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 18.29,94.71 C 18.29,88.71 44.37,83.11 125.21,83.11 C 206.06,83.11 231.37,88.71 231.37,94.71 C 231.37,100.71 206.06,106.31 125.21,106.31 C 44.37,106.31 18.29,100.71 18.29,94.71 Z" fill="#391A51" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 20.79,94.71 C 20.79,89.21 46.37,84.11 125.21,84.11 C 204.06,84.11 228.87,89.21 228.87,94.71 H 20.79 Z" stroke="#8B37B5" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 11.75,101.01 V 105.81 C 11.75,112.11 32.62,119.91 125.21,119.91 C 217.8,119.91 237.71,112.11 237.71,105.81 V 100.11 C 237.71,106.41 217.8,114.21 125.21,114.21 C 32.62,114.21 11.75,106.41 11.75,101.01 Z" fill="#391A51" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 12.2,101.01 V 103.71 C 12.2,109.61 33.22,117.01 125.21,117.01 C 217.21,117.01 237.71,109.61 237.71,103.71 V 100.11 C 237.71,106.01 217.21,113.41 125.21,113.41 C 33.22,113.41 12.2,106.01 12.2,101.01 Z" fill="#480099" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 12.2,102.21 C 12.2,107.71 33.22,114.91 125.21,114.91 C 217.21,114.91 237.71,107.71 237.71,102.21 V 103.71 C 237.71,109.61 217.21,117.01 125.21,117.01 C 33.22,117.01 12.2,109.61 12.2,103.71 V 102.21 Z" fill="#48D9FF" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Support columns + detail */}
    <path d="M 22.84,112.91 L 32.02,121.21 L 33.72,125.91 L 33.97,137.81 L 36.87,145.61 L 43.11,140.01 L 43.31,131.01 L 50.71,137.81 L 54.21,144.81 L 60.15,149.21 L 74.28,157.61 L 85.28,162.11 L 103.82,164.01 L 113.91,167.21 H 138.75 L 153.72,164.01 L 164.31,159.31 L 173.51,156.91 L 183.61,152.51 L 197.73,137.81 L 201.49,134.61 L 213.78,129.71 L 216.68,126.61 L 217.38,123.21 L 226.02,114.21 C 210.33,119.11 186.96,120.81 125.21,120.81 C 68.18,120.81 43.71,118.91 22.84,112.91 Z" fill="#3D2360" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 27.92,117.01 L 35.12,122.71 L 47.91,127.11 L 59.6,137.81 H 74.28 L 103.82,141.01 L 113.91,147.01 H 138.75 L 153.72,144.81 L 169.39,142.41 L 185.06,138.51 L 198.64,132.91 L 209.38,127.11 L 218.48,123.21 L 220.78,120.31 C 208.08,123.81 193.05,124.61 125.21,124.61 C 69.13,124.61 50.01,122.71 27.92,117.01 Z" fill="#4B2875" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 38.42,117.01 L 46.12,117.61 L 57.8,132.91 L 65.33,137.81 L 74.83,138.51 L 85.28,139.71 L 103.82,141.61 L 106.32,143.41 L 113.91,144.01 H 136.89 L 139.75,142.41 L 153.72,141.61 L 169.39,139.71 L 179.75,137.81 L 187.11,137.11 L 194.98,131.01 L 203.19,117.61 L 209.38,116.11 L 203.19,117.01 L 192.03,132.21 L 187.11,134.61 L 156.11,137.81 L 148.62,139.11 L 138.75,139.71 H 112.4 L 107.71,137.81 L 77.79,135.31 L 65.33,133.71 L 52.11,120.31 L 53.38,118.61 L 46.12,117.61 L 38.42,117.01 Z" fill="#40276A" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Leg columns */}
    <path d="M 59.6,137.81 L 60.15,172.41 H 71.48 L 72.68,138.51 L 65.33,137.81 H 59.6 Z" fill="url(#parallax-paint0)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 85.28,141.01 V 172.41 H 103.82 V 141.61 L 97.22,142.41 L 91.78,141.01 H 85.28 Z" fill="url(#parallax-paint1)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 113.91,137.81 V 172.41 H 135.44 V 137.81 L 125.21,137.11 L 113.91,137.81 Z" fill="url(#parallax-paint2)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 162.11,141.01 L 162.71,172.41 H 175.66 L 176.23,140.01 L 171.48,141.01 L 166.69,138.51 L 162.11,141.01 Z" fill="url(#parallax-paint3)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 181.15,141.01 L 183.61,144.01 V 166.01 L 185.31,169.31 L 187.11,166.01 V 137.81 L 182.85,138.51 L 181.15,141.01 Z" fill="url(#parallax-paint4)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 33.72,127.11 L 34.12,138.51 L 36.87,144.81 L 42.81,139.11 L 42.51,131.01 L 33.72,127.11 Z" fill="url(#parallax-paint5)" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Structural panels */}
    <path d="M 110.01,124.61 L 109.61,129.71 L 103.07,129.11 L 105.92,134.61 L 110.01,138.51 H 138.75 V 134.61 L 143.94,130.11 V 129.11 H 135.9 L 135.44,124.61 H 110.01 Z" fill="#6334A5" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 162.11,124.61 L 161.06,129.71 L 181.7,128.01 V 124.61 H 162.11 Z" fill="#3476BA" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 157.81,123.21 L 153.72,139.71 L 150.12,140.01 L 154.42,123.21 H 157.81 Z" fill="#5A2A70" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 139.75,124.61 L 138.75,129.11 H 143.94 L 138.75,134.61 V 138.51 L 147.69,124.61 H 139.75 Z" fill="#226699" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 85.28,123.91 L 86.68,131.01 H 93.02 L 91.78,124.61 L 85.28,123.91 Z" fill="#004787" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 94.47,124.61 L 96.12,131.01 H 101.67 L 99.22,124.61 H 94.47 Z" fill="#004787" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Center V detail */}
    <path d="M 112.4,137.11 L 114.61,139.71 L 116.91,150.71 L 123.71,154.11 L 130.41,147.61 L 132.01,137.11 H 129.15 L 126.75,149.21 L 123.71,151.61 L 119.01,137.11 H 112.4 Z" fill="#327899" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 117.66,137.81 L 120.06,147.01 L 122.76,149.21 L 126.75,145.61 L 128.4,137.81 H 126.75 L 124.21,147.01 L 122.76,147.61 L 120.06,137.81 H 117.66 Z" fill="#48D9FF" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Structural lines */}
    <path d="M 110.01,129.71 H 138.75" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 178.7,138.51 L 181.15,141.61" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 166.69,139.71 L 171.48,143.41 L 176.23,139.71" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 100.12,142.41 L 95.87,146.21 L 90.18,142.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 120.71,149.21 V 172.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 125.21,151.61 V 171.91" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 134.1,156.91 L 127.91,170.31" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 169.39,143.41 V 172.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 93.02,145.61 V 172.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 97.22,146.21 V 172.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 65.33,139.71 V 172.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 68.78,142.41 L 67.13,146.21 V 150.71 L 71.48,147.01" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 67.13,156.91 L 70.78,154.11" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M 36.87,129.11 V 139.71" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Edge glow lines */}
    <path className="wire-accent" d="M 40.71,109.91 V 112.41" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 60.15,112.41 V 113.91" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 71.48,113.41 L 71.78,116.61" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 135.44,116.11 V 117.01" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 174.91,114.21 L 175.66,117.01" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 208.13,110.11 L 208.63,112.41" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 179.05,125.31 L 179.75,127.11" stroke="#48D9FF" strokeWidth="0.75" strokeMiterlimit="10"/>
    {/* Platform surface strokes */}
    <path d="M 32.02,87.91 L 39.82,89.61" stroke="#8B37B5" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 206.93,88.61 L 215.13,90.11" stroke="#8B37B5" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Top surface */}
    <path d="M 43.31,89.61 C 43.31,86.01 66.08,83.11 125.21,83.11 C 184.35,83.11 206.93,86.01 206.93,89.61 C 206.93,93.21 184.35,96.11 125.21,96.11 C 66.08,96.11 43.31,93.21 43.31,89.61 Z" fill="#391A51" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 67.13,93.71 C 67.13,90.41 82.11,87.71 125.21,87.71 C 168.32,87.71 182.85,90.41 182.85,93.71 C 182.85,97.01 168.32,99.71 125.21,99.71 C 82.11,99.71 67.13,97.01 67.13,93.71 Z" fill="#9836B5" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path className="wire-surface" d="M 75.13,92.91 C 75.13,90.11 87.23,87.81 125.21,87.81 C 163.2,87.81 174.86,90.11 174.86,92.91 C 174.86,95.71 163.2,98.01 125.21,98.01 C 87.23,98.01 75.13,95.71 75.13,92.91 Z" fill="#48B9D9" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path className="wire-accent" d="M 100.12,92.91 C 100.12,91.01 107.81,89.41 125.21,89.41 C 142.62,89.41 148.62,91.01 148.62,92.91 C 148.62,94.81 142.62,96.41 125.21,96.41 C 107.81,96.41 100.12,94.81 100.12,92.91 Z" fill="#B4F5F0" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Bottom structure lines */}
    <path d="M 124.71,100.11 V 104.41" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 184.61,104.41 L 185.71,106.31" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    <path d="M 187.11,111.71 V 112.91" stroke="#391A51" strokeWidth="0.5" strokeMiterlimit="10"/>
    {/* Collar ring */}
    <path d="M 174.86,113.41 L 175.66,117.61 V 120.81 C 164.21,121.71 150.12,122.01 125.21,122.01 C 104.82,122.01 90.18,121.61 78.29,120.81 L 72.68,120.31 L 71.78,114.21 C 83.08,115.31 98.92,116.11 125.21,116.11 C 149.12,116.11 164.81,115.11 174.86,113.41 Z" fill="#6334A5" stroke="black" strokeWidth="0.5" strokeMiterlimit="10"/>
  </svg>
);

const LAYERS = [Layer1, Layer2, Layer3, Layer4];

/**
 * Compute where the SVG platform naturally falls in viewport Y,
 * then compute where the game ground line is (or should be),
 * and return the delta as a base offset for the stage layers.
 */
function computeStageOffset(): number {
  const vh = window.innerHeight;
  const vw = window.innerWidth;

  // Layer div is 120% of viewport, offset by -10%
  const layerW = 1.2 * vw;
  const layerH = 1.2 * vh;

  // With preserveAspectRatio="xMidYMid slice", SVG covers the container
  let platYInLayer: number;
  if (layerW / layerH > SVG_ASPECT) {
    // Viewport wider than SVG — scale to width, center vertically
    const scale = layerW / SVG_W;
    const renderedH = SVG_H * scale;
    const yOff = (layerH - renderedH) / 2;
    platYInLayer = yOff + (SVG_PLAT_Y / SVG_H) * renderedH;
  } else {
    // Viewport taller — scale to height, SVG fills vertically
    platYInLayer = (SVG_PLAT_Y / SVG_H) * layerH;
  }

  // Platform Y in viewport coords (layer starts at -10% of vh)
  const platYViewport = -0.1 * vh + platYInLayer;

  // Target: where should the platform be?
  const crt = document.querySelector('.crt-screen');
  let targetY: number;
  if (crt) {
    const rect = crt.getBoundingClientRect();
    targetY = rect.top + GROUND_FRAC * rect.height;
  } else {
    // No CRT on screen — place platform at ~63% of viewport (natural look)
    targetY = vh * 0.63;
  }

  return targetY - platYViewport;
}

export default function ParallaxBackground() {
  const containerRef = useRef<HTMLDivElement>(null);
  const baseOffsetRef = useRef(0);
  const lastAlignRef = useRef(0);

  const applyTransforms = useCallback((cx: number, cy: number, zoom: number) => {
    const el = containerRef.current;
    if (!el) return;
    const dx = cx - REST_X;
    const dy = cy - REST_Y;
    const pxPerUnit = el.clientWidth / 340;
    const offset = baseOffsetRef.current;

    for (let i = 0; i < RATES.length; i++) {
      const rate = RATES[i];
      const tx = -dx * pxPerUnit * rate;
      const ty = dy * pxPerUnit * rate + (i >= 2 ? offset : 0);
      const s = 1 + (zoom - 1) * rate * 0.5;
      el.style.setProperty(`--l${i}-tx`, `${tx}px`);
      el.style.setProperty(`--l${i}-ty`, `${ty}px`);
      el.style.setProperty(`--l${i}-s`, `${s}`);
    }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Initial alignment (defer one frame so layout is settled)
    requestAnimationFrame(() => {
      baseOffsetRef.current = computeStageOffset();
      applyTransforms(REST_X, REST_Y, 1);
    });

    const onResize = () => {
      baseOffsetRef.current = computeStageOffset();
      applyTransforms(REST_X, REST_Y, 1);
    };
    window.addEventListener('resize', onResize);

    setCameraListener((cx, cy, zoom) => {
      // Lazy re-check alignment every 2s (catches route changes where CRT appears/disappears)
      const now = performance.now();
      if (now - lastAlignRef.current > 2000) {
        lastAlignRef.current = now;
        baseOffsetRef.current = computeStageOffset();
      }
      applyTransforms(cx, cy, zoom);
    });

    return () => {
      window.removeEventListener('resize', onResize);
      setCameraListener(null);
    };
  }, [applyTransforms]);

  return (
    <div ref={containerRef} className="parallax-bg">
      {LAYERS.map((LayerComp, i) => (
        <div
          key={i}
          className="parallax-layer"
          style={{
            transform: `translate3d(var(--l${i}-tx, 0px), var(--l${i}-ty, 0px), 0) scale(var(--l${i}-s, 1))`,
          }}
        >
          <LayerComp />
        </div>
      ))}
    </div>
  );
}
