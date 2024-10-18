const socket = io();

const analyzeBtn = document.getElementById('analyzeBtn');
const analysisProgress = document.getElementById('analysisProgress');
const progressBar = document.querySelector('.progress-bar');
const progressText = document.querySelector('.progress-text');
const analysisResult = document.getElementById('analysisResult');

analyzeBtn.addEventListener('click', () => {
    analyzeBtn.style.display = 'none';
    analysisProgress.style.display = 'block';
    analysisResult.innerHTML = '';
    socket.emit('start_analysis');
});

socket.on('analysis_progress', (data) => {
    const progress = data.progress;
    gsap.to(progressBar, {
        width: `${progress}%`,
        duration: 0.5,
        ease: 'power2.out'
    });
    progressText.textContent = `${progress}%`;
});

socket.on('analysis_complete', (data) => {
    gsap.to(analysisProgress, {
        opacity: 0,
        duration: 1,
        delay: 1,
        onComplete: () => {
            analysisProgress.style.display = 'none';
            displayResults(data.results);
        }
    });
});

function displayResults(results) {
    analysisResult.innerHTML = '';
    if (results.length === 0) {
        analysisResult.innerHTML = '<p>Tidak ada sinyal yang ditemukan.</p>';
        return;
    }
    results.forEach((result, index) => {
        const resultElement = document.createElement('div');
        resultElement.classList.add('result-item');
        resultElement.innerHTML = `
            <h3>${result.symbol}</h3>
            <p>Sinyal: ${result.signal}</p>
            <p>Kekuatan: ${result.strength.toFixed(2)}%</p>
            <p>Indikator Naik: ${result.up_signals}</p>
            <p>Indikator Turun: ${result.down_signals}</p>
            <p>Konsensus Terbanyak: ${result.max_consensus}</p>
            <p>Sentiment: ${result.sentiment}</p>
            <p>Support: ${result.support}</p>
            <p>Resistance: ${result.resistance}</p>
        `;
        gsap.from(resultElement, {
            opacity: 0,
            y: 50,
            duration: 0.5,
            delay: index * 0.1
        });
        analysisResult.appendChild(resultElement);
    });
}
