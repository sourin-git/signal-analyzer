// SECTION 1 — Signal Math

// Generates an array signal representing standard wave types
function generateSignal(type, N, freq) {
    const t = Array.from({ length: N }, (_, i) => i / N);
    if (type === 'sine') return t.map(x => Math.sin(2 * Math.PI * freq * x));
    if (type === 'square') return t.map(x => Math.sign(Math.sin(2 * Math.PI * freq * x)));
    if (type === 'triangle') return t.map(x => 2 * Math.abs(2 * (x * freq - Math.floor(x * freq + 0.5))) - 1);
    if (type === 'ecg') return t.map(x => { // Emulates basic P-QRS-T complex
        let p = (x * freq) % 1.0;
        if (0.1 <= p && p < 0.15) return 0.25 * Math.sin(Math.PI * (p - 0.1) / 0.05);
        if (0.3 <= p && p < 0.35) return -0.2 * Math.sin(Math.PI * (p - 0.3) / 0.05);
        if (0.35 <= p && p <= 0.4) return 1.0 * Math.sin(Math.PI * (p - 0.35) / 0.05);
        if (0.4 < p && p <= 0.45) return -0.3 * Math.sin(Math.PI * (p - 0.4) / 0.05);
        if (0.6 <= p && p < 0.7) return 0.35 * Math.sin(Math.PI * (p - 0.6) / 0.1);
        return 0;
    });
    return t.map(() => 0);
}

// Maps a uniform random number offset over the sequence
function addNoise(signal, level) {
    return signal.map(val => val + (Math.random() * 2 - 1) * level);
}

// Applies flat convolution against local rolling neighbors limits
function movingAverage(signal, w) {
    const half = Math.floor(w / 2);
    return signal.map((_, i) => {
        let sum = 0, count = 0;
        for (let j = -half; j <= half; j++) {
            if (i + j >= 0 && i + j < signal.length) {
                sum += signal[i + j];
                count++;
            }
        }
        return sum / count;
    });
}

// Computes normally distributed decaying weights scaling locally
function gaussianSmooth(signal, w) {
    const half = Math.floor(w / 2);
    const sigma = w / 6;
    let weights = [], wSum = 0;
    for (let j = -half; j <= half; j++) {
        let wt = Math.exp(-(j * j) / (2 * sigma * sigma));
        weights.push(wt);
        wSum += wt;
    }
    return signal.map((_, i) => {
        let sum = 0, count = 0;
        for (let j = -half; j <= half; j++) {
            if (i + j >= 0 && i + j < signal.length) {
                sum += signal[i + j] * weights[j + half];
                count += weights[j + half];
            }
        }
        return sum / count;
    });
}

// Performs an algebraic sort of the window block selecting the center value
function medianFilter(signal, w) {
    const half = Math.floor(w / 2);
    return signal.map((_, i) => {
        let win = [];
        for (let j = -half; j <= half; j++) {
            if (i + j >= 0 && i + j < signal.length) win.push(signal[i + j]);
        }
        win.sort((a, b) => a - b);
        return win[Math.floor(win.length / 2)];
    });
}

// Cross-verifies difference mapping determining percentage and relative distances
function computeMetrics(clean, noisy, processed) {
    let mseBefore = 0, mseAfter = 0;
    for (let i = 0; i < clean.length; i++) {
        mseBefore += Math.pow(clean[i] - noisy[i], 2);
        mseAfter += Math.pow(clean[i] - processed[i], 2);
    }
    const rmseBefore = Math.sqrt(mseBefore / clean.length);
    const rmseAfter = Math.sqrt(mseAfter / clean.length);
    const pctReduced = rmseBefore === 0 ? 0 : ((rmseBefore - rmseAfter) / rmseBefore) * 100;
    return { rmseBefore, rmseAfter, pctReduced };
}

// SECTION 2 — Plotly Chart

// Executes an embedded Plotly chart replacing data array definitions silently mapping over DOM #plotly-chart
function renderPlotly(clean, noisy, processed) {
    const t = Array.from({ length: clean.length }, (_, i) => i);
    const traceClean = { x: t, y: clean, name: 'Ideal', line: { color: 'cyan', dash: 'dash' } };
    const traceNoisy = { x: t, y: noisy, name: 'Noisy', line: { color: 'orange' }, opacity: 0.6 };
    const traceProc = { x: t, y: processed, name: 'Processed', line: { color: 'green' } };
    const layout = { 
        template: 'plotly_dark', 
        paper_bgcolor: 'transparent', 
        plot_bgcolor: 'transparent', 
        margin: { t: 40, r: 20, b: 40, l: 40 }, 
        showlegend: false 
    };
    Plotly.newPlot('plotly-chart', [traceClean, traceNoisy, traceProc], layout, { displayModeBar: false });
}

// SECTION 3 — Three.js 3D Spectrum
let threeScene, threeCamera, threeRenderer, spectrumGroup;
let isDragging = false, prevMouse = { x: 0, y: 0 };

// Implements a discrete Fourier transform projecting heights onto independent z-axis grouped BoxGeometries
function render3DSpectrum(noisy, processed) {
    if (!threeScene) initThreeJS();
    
    // Compute DFT bounds limits directly scaling up 64 nodes maximum
    const bins = Math.min(64, noisy.length);
    let noisyFFT = new Array(bins).fill(0);
    let procFFT  = new Array(bins).fill(0);
    
    for (let k = 0; k < bins; k++) {
        let realN = 0, imagN = 0, realP = 0, imagP = 0;
        for (let n = 0; n < noisy.length; n++) {
            let angle = (2 * Math.PI * k * n) / noisy.length;
            realN += noisy[n] * Math.cos(angle);
            imagN -= noisy[n] * Math.sin(angle);
            realP += processed[n] * Math.cos(angle);
            imagP -= processed[n] * Math.sin(angle);
        }
        noisyFFT[k] = Math.sqrt(realN*realN + imagN*imagN) / noisy.length;
        procFFT[k] = Math.sqrt(realP*realP + imagP*imagP) / processed.length;
    }

    while(spectrumGroup.children.length > 0){ spectrumGroup.remove(spectrumGroup.children[0]); }
    
    const geometry = new THREE.BoxGeometry(0.8, 1, 0.8);
    const matOrange = new THREE.MeshBasicMaterial({ color: 0xDB6D28 });
    const matGreen = new THREE.MeshBasicMaterial({ color: 0x3FB950 });

    for (let i = 0; i < bins; i++) {
        // Row 1 (z=0): Noisy orange bars
        let hN = Math.max(0.1, noisyFFT[i] * 5);
        const meshN = new THREE.Mesh(geometry, matOrange);
        meshN.scale.y = hN;
        meshN.position.set(i - bins/2, hN/2, 0);
        spectrumGroup.add(meshN);
        
        // Row 2 (z=3): Processed green bars
        let hP = Math.max(0.1, procFFT[i] * 5);
        const meshP = new THREE.Mesh(geometry, matGreen);
        meshP.scale.y = hP;
        meshP.position.set(i - bins/2, hP/2, 3);
        spectrumGroup.add(meshP);
    }
}

// Spawns WebGL rendering pipeline attaching custom dragging properties resolving view size recalculations seamlessly
function initThreeJS() {
    const container = document.getElementById('three-canvas');
    threeScene = new THREE.Scene();
    threeCamera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight || 1.77, 0.1, 1000);
    threeCamera.position.set(0, 15, 30);
    threeCamera.lookAt(0, 0, 0);

    threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    threeRenderer.setSize(container.clientWidth || 800, container.clientHeight || 450);
    container.appendChild(threeRenderer.domElement);

    spectrumGroup = new THREE.Group();
    threeScene.add(spectrumGroup);

    // Manual orbit controls via mouse
    container.addEventListener('mousedown', (e) => { isDragging = true; prevMouse = { x: e.offsetX, y: e.offsetY }; });
    document.addEventListener('mouseup', () => isDragging = false);
    container.addEventListener('mousemove', (e) => {
        if (isDragging) {
            let dx = e.offsetX - prevMouse.x;
            let dy = e.offsetY - prevMouse.y;
            spectrumGroup.rotation.y += dx * 0.01;
            spectrumGroup.rotation.x += dy * 0.01;
            prevMouse = { x: e.offsetX, y: e.offsetY };
        }
    });

    const animate = function () {
        requestAnimationFrame(animate);
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        // Dynamic resize for hidden-by-default container scaling corrections
        if (cw > 0 && ch > 0 && (threeRenderer.domElement.width !== cw || threeRenderer.domElement.height !== ch)) {
            threeRenderer.setSize(cw, ch);
            threeCamera.aspect = cw / ch;
            threeCamera.updateProjectionMatrix();
        }
        threeRenderer.render(threeScene, threeCamera);
    };
    animate();
}

// SECTION 4 — Compare Table

// Interrogates strictly 3 fundamental filters extracting optimal percentages rewriting innerHTML table sequentially
function buildCompareTable(clean, noisy) {
    const w = parseInt(document.getElementById('window-size').value, 10);
    const filters = [
        { name: 'Moving Average', data: movingAverage(noisy, w) },
        { name: 'Gaussian Smooth', data: gaussianSmooth(noisy, w) },
        { name: 'Median Filter', data: medianFilter(noisy, w) }
    ];
    
    let results = filters.map(f => {
        const m = computeMetrics(clean, noisy, f.data);
        return { name: f.name, rmse: m.rmseAfter, pct: m.pctReduced };
    });
    
    // Scan locally for best filter node
    let bestIdx = 0;
    for(let i = 1; i < results.length; i++) {
        if(results[i].pct > results[bestIdx].pct) bestIdx = i;
    }
    
    const tbody = document.getElementById('compare-tbody');
    tbody.innerHTML = '';
    
    results.forEach((r, i) => {
        const isBest = i === bestIdx;
        const tr = document.createElement('tr');
        if(isBest) tr.style.color = 'var(--green)';
        tr.innerHTML = `
            <td>${r.name}</td>
            <td class="mono">${r.rmse.toFixed(4)}</td>
            <td class="mono">${r.pct.toFixed(2)}%</td>
            <td>${isBest ? '★ Best' : ''}</td>
        `;
        tbody.appendChild(tr);
    });
}

// SECTION 5 — Event listeners

// Invokes core algorithms extracting static identifiers replacing active visual data matrices collectively
function processAndUpdate() {
    const type = document.getElementById('signal-type').value;
    const noiseLevel = parseFloat(document.getElementById('noise-level').value);
    const filterName = document.getElementById('filter-method').value;
    const windowSize = parseInt(document.getElementById('window-size').value, 10);
    
    document.getElementById('noise-val').textContent = noiseLevel.toFixed(1);
    
    const clean = generateSignal(type, 256, 3);
    const noisy = addNoise(clean, noiseLevel);
    
    let processed;
    if (filterName === 'moving_average' || filterName === 'savitzky_golay') processed = movingAverage(noisy, windowSize); // Fallback SG
    else if (filterName === 'gaussian_smooth') processed = gaussianSmooth(noisy, windowSize);
    else if (filterName === 'median_filter') processed = medianFilter(noisy, windowSize);
    else processed = movingAverage(noisy, windowSize);
    
    const metrics = computeMetrics(clean, noisy, processed);
    
    document.getElementById('stat-rmse-before').textContent = metrics.rmseBefore.toFixed(4);
    document.getElementById('stat-rmse-after').textContent = metrics.rmseAfter.toFixed(4);
    document.getElementById('stat-noise-pct').textContent = metrics.pctReduced.toFixed(2) + '%';
    
    renderPlotly(clean, noisy, processed);
    render3DSpectrum(noisy, processed);
    buildCompareTable(clean, noisy);
}

// Locks UI event loops ensuring standard configuration loads default views upon initialization seamlessly
window.addEventListener('load', () => {
    document.getElementById('apply-btn').addEventListener('click', processAndUpdate);
    document.getElementById('noise-level').addEventListener('input', (e) => {
        document.getElementById('noise-val').textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    processAndUpdate();
});
