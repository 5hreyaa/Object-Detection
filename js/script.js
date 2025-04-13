document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const assessBtn = document.getElementById('assessBtn');
    const anprBtn = document.getElementById('anprBtn');
    const downloadPdf = document.getElementById('downloadPdf');
    const resultsSection = document.getElementById('resultsSection');
    const uploadTab = document.getElementById('uploadTab');
    const webcamTab = document.getElementById('webcamTab');
    const uploadSection = document.getElementById('uploadSection');
    const webcamSection = document.getElementById('webcamSection');
    const webcamStream = document.getElementById('webcamStream');
    const captureBtn = document.getElementById('captureBtn');
    const processingOverlay = document.getElementById('processingOverlay');
    const plateResult = document.getElementById('plateResult');
    const plateText = document.getElementById('plateText');
    const damageContainer = document.getElementById('damageContainer');
    const resultImageContainer = document.getElementById('resultImageContainer');
    const damageTags = document.getElementById('damageTags');
    const totalCost = document.getElementById('totalCost');
    const laborCost = document.getElementById('laborCost');
    const partsCost = document.getElementById('partsCost');
    const repairTime = document.getElementById('repairTime');
    const costChart = document.getElementById('costChart');
    const noResultsMessage = document.getElementById('noResultsMessage');
    
    let currentFile = null;
    let webcamActive = false;
    let stream = null;
    
    // Initialize
    resultsSection.classList.add('visible');
    
    // Event Listeners
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('active');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    uploadTab.addEventListener('click', () => {
        uploadTab.classList.add('active');
        webcamTab.classList.remove('active');
        uploadSection.style.display = 'block';
        webcamSection.style.display = 'none';
        stopWebcam();
    });
    
    webcamTab.addEventListener('click', () => {
        webcamTab.classList.add('active');
        uploadTab.classList.remove('active');
        uploadSection.style.display = 'none';
        webcamSection.style.display = 'block';
        startWebcam();
    });
    
    assessBtn.addEventListener('click', () => {
        if (currentFile) {
            processDamageAssessment();
        } else if (webcamActive) {
            captureWebcam();
            setTimeout(() => {
                processDamageAssessment();
            }, 500);
        } else {
            alert('Please upload an image or use webcam first');
        }
    });
    
    anprBtn.addEventListener('click', () => {
        if (currentFile) {
            processANPR();
        } else if (webcamActive) {
            captureWebcam();
            setTimeout(() => {
                processANPR();
            }, 500);
        } else {
            alert('Please upload an image or use webcam first');
        }
    });
    
    captureBtn.addEventListener('click', captureWebcam);
    
    downloadPdf.addEventListener('click', generatePDF);
    
    // Functions
    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            dropZone.innerHTML = `
                <img src="${e.target.result}" alt="Uploaded image" style="max-height: 120px; max-width: 100%;">
                <p class="mt-2 mb-0">${file.name}</p>
            `;
        };
        
        reader.readAsDataURL(file);
    }
    
    function startWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(videoStream) {
                    stream = videoStream;
                    webcamStream.srcObject = videoStream;
                    webcamActive = true;
                    document.querySelector('.webcam-container').style.display = 'block';
                })
                .catch(function(error) {
                    console.error('Webcam error: ', error);
                    alert('Could not access webcam');
                });
        } else {
            alert('Your browser does not support webcam access');
        }
    }
    
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamActive = false;
        }
    }
    
    function captureWebcam() {
        if (!webcamActive) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = webcamStream.videoWidth;
        canvas.height = webcamStream.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamStream, 0, 0);
        
        // Convert canvas to file
        canvas.toBlob((blob) => {
            currentFile = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
            
            // Display captured image
            const reader = new FileReader();
            reader.onload = (e) => {
                dropZone.innerHTML = `
                    <img src="${e.target.result}" alt="Captured image" style="max-height: 120px; max-width: 100%;">
                    <p class="mt-2 mb-0">Webcam Capture</p>
                `;
            };
            reader.readAsDataURL(currentFile);
        });
    }
    
    function showProcessing() {
        processingOverlay.style.display = 'flex';
    }
    
    function hideProcessing() {
        processingOverlay.style.display = 'none';
    }
    
    function processDamageAssessment() {
        if (!currentFile) return;
        
        showProcessing();
        
        // Create a form data object to send the image
        const formData = new FormData();
        formData.append('image', currentFile);
        
        // In a real application, you would send this to your backend
        // For demo purposes, we'll simulate a response
        setTimeout(() => {
            hideProcessing();
            
            // Example damage detection results
            const mockResponse = {
                image_data: currentFile.type.includes('image') ? URL.createObjectURL(currentFile) : '/path/to/result/image.jpg',
                report: {
                    damage_summary: [
                        { damage_type: 'Scratch', severity: 'moderate', confidence: 0.87 },
                        { damage_type: 'Dent', severity: 'severe', confidence: 0.92 },
                        { damage_type: 'Broken Bumper', severity: 'severe', confidence: 0.78 }
                    ],
                    cost_breakdown: {
                        total: 12500,
                        parts: 7500,
                        labor: 5000
                    },
                    repair_time_estimate: 6,
                    vehicle_info: {
                        make: "",
                        model: "",
                        year: ""
                    }
                }
            };
            
            displayDamageResults(mockResponse);
        }, 2000);
    }
    
    function processANPR() {
        if (!currentFile) return;
        
        showProcessing();
        
        // Create a form data object to send the image
        const formData = new FormData();
        formData.append('image', currentFile);
        
        // Simulate ANPR processing
        setTimeout(() => {
            hideProcessing();
            
            // Example ANPR results
            const mockResponse = {
                image_data: currentFile.type.includes('image') ? URL.createObjectURL(currentFile) : '/path/to/result/image.jpg',
                plate_text: "MH12DE5678",
                plate_bbox: [100, 200, 300, 250]
            };
            
            displayANPRResults(mockResponse);
        }, 1500);
    }
    
    function displayDamageResults(response) {
        // Reset display
        resultImageContainer.innerHTML = '';
        damageTags.innerHTML = '';
        plateResult.style.display = 'none';
        noResultsMessage.style.display = 'none';
        
        // Display result image
        const img = document.createElement('img');
        img.src = response.image_data;
        img.classList.add('result-image');
        img.alt = 'Damage Assessment Result';
        resultImageContainer.appendChild(img);
        
        // Show damage container
        damageContainer.style.display = 'block';
        
        // Add damage tags
        response.report.damage_summary.forEach(damage => {
            damageTags.innerHTML += `
                <span class="damage-tag">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    ${damage.damage_type} (${(damage.confidence * 100).toFixed(0)}%)
                </span>
            `;
        });
        
        // Update cost breakdown
        totalCost.textContent = `₹${response.report.cost_breakdown.total.toLocaleString()}`;
        laborCost.textContent = `₹${response.report.cost_breakdown.labor.toLocaleString()}`;
        partsCost.textContent = `₹${response.report.cost_breakdown.parts.toLocaleString()}`;
        repairTime.textContent = `${response.report.repair_time_estimate} days`;
        
        // Create cost breakdown chart
        Plotly.newPlot(costChart, [{
            values: [response.report.cost_breakdown.parts, response.report.cost_breakdown.labor],
            labels: ['Parts', 'Labor'],
            type: 'pie',
            marker: {
                colors: ['#3498db', '#2ecc71']
            },
            textinfo: 'percent',
            hole: 0.4
        }], {
            height: 200,
            width: 200,
            margin: {l: 0, r: 0, b: 0, t: 0},
            showlegend: false
        });
        
        // Show download button
        downloadPdf.style.display = 'block';
    }
    
    function displayANPRResults(response) {
        // Reset display
        resultImageContainer.innerHTML = '';
        damageContainer.style.display = 'none';
        noResultsMessage.style.display = 'none';
        
        // Display result image
        const img = document.createElement('img');
        img.src = response.image_data;
        img.classList.add('result-image');
        img.alt = 'License Plate Recognition Result';
        resultImageContainer.appendChild(img);
        
        // Show plate result
        plateResult.style.display = 'flex';
        plateText.textContent = response.plate_text;
        
        // Show download button
        downloadPdf.style.display = 'block';
    }
    
    function generatePDF() {
        showProcessing();
        
        // In a real application, this would send a request to generate a PDF report
        // For demo purposes, we'll simulate this
        setTimeout(() => {
            hideProcessing();
            
            // Create a fake download
            const link = document.createElement('a');
            link.href = '#';
            link.download = 'VisionX_Report.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // Show success message
            const successAlert = document.createElement('div');
            successAlert.className = 'alert alert-success animate__animated animate__fadeIn';
            successAlert.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                <strong>Success!</strong> Your report has been downloaded.
            `;
            document.querySelector('.main-container').prepend(successAlert);
            
            // Remove alert after 3 seconds
            setTimeout(() => {
                successAlert.classList.remove('animate__fadeIn');
                successAlert.classList.add('animate__fadeOut');
                setTimeout(() => successAlert.remove(), 500);
            }, 3000);
        }, 2000);
    }
});