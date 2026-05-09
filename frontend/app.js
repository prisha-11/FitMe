const API_BASE = 'http://localhost:8083/api';

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/dashboard-stats`);
        const data = await res.json();
        
        document.getElementById('stat-products').innerText = data.total_products.toLocaleString();
        document.getElementById('stat-price').innerText = `$${data.avg_price}`;
        document.getElementById('stat-sales').innerText = data.total_sales.toLocaleString();
        document.getElementById('stat-risk').innerText = data.high_risk_count.toLocaleString();
    } catch (e) {
        console.error("Error fetching stats:", e);
    }
}

let categoryChartInstance = null;
let riskChartInstance = null;

async function renderCategorySalesChart() {
    try {
        const res = await fetch(`${API_BASE}/charts/category-sales`);
        const data = await res.json();

        var options = {
            series: [{
                name: 'Total Sales',
                data: data.series
            }],
            chart: {
                type: 'bar',
                height: 320,
                toolbar: { show: false },
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800,
                    animateGradually: {
                        enabled: true,
                        delay: 150
                    },
                    dynamicAnimation: {
                        enabled: true,
                        speed: 350
                    }
                }
            },
            plotOptions: {
                bar: {
                    borderRadius: 6,
                    columnWidth: '45%',
                    distributed: true,
                }
            },
            dataLabels: { enabled: false },
            legend: { show: false },
            xaxis: {
                categories: data.labels,
                labels: { style: { colors: '#94a3b8', fontSize: '12px' } },
                axisBorder: { show: false },
                axisTicks: { show: false }
            },
            yaxis: {
                labels: { style: { colors: '#94a3b8' } }
            },
            grid: {
                borderColor: 'rgba(255,255,255,0.05)',
                strokeDashArray: 4,
            },
            colors: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e'],
            tooltip: {
                theme: 'dark'
            }
        };

        if (categoryChartInstance) {
            categoryChartInstance.destroy();
        }
        categoryChartInstance = new ApexCharts(document.querySelector("#chart-category-sales"), options);
        categoryChartInstance.render();
    } catch (e) {
        console.error("Error rendering category sales chart", e);
    }
}

async function renderRiskDistributionChart() {
    try {
        const res = await fetch(`${API_BASE}/charts/risk-distribution`);
        const data = await res.json();

        var options = {
            series: data.series,
            chart: {
                type: 'donut',
                height: 320,
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800,
                }
            },
            labels: data.labels,
            colors: ['#ef4444', '#f59e0b', '#10b981'], // Red, Yellow, Green for High, Med, Low
            plotOptions: {
                pie: {
                    donut: {
                        size: '70%',
                        labels: {
                            show: true,
                            name: { color: '#94a3b8' },
                            value: { color: '#f8f9fa', fontSize: '24px', fontWeight: 600 }
                        }
                    }
                }
            },
            dataLabels: { enabled: false },
            legend: {
                position: 'bottom',
                labels: { colors: '#94a3b8' }
            },
            stroke: { show: false },
            tooltip: { theme: 'dark' }
        };

        if (riskChartInstance) {
            riskChartInstance.destroy();
        }
        riskChartInstance = new ApexCharts(document.querySelector("#chart-risk-dist"), options);
        riskChartInstance.render();
    } catch (e) {
        console.error("Error rendering risk distribution chart", e);
    }
}

async function fetchMLEvaluation() {
    try {
        const res = await fetch(`${API_BASE}/models/evaluation`);
        const data = await res.json();

        document.getElementById('ml-acc').innerText = `${data.classification_accuracy}%`;
        document.getElementById('ml-r2').innerText = data.regression_r2;

        // Animate progress bars
        setTimeout(() => {
            document.getElementById('ml-acc-bar').style.width = `${data.classification_accuracy}%`;
            document.getElementById('ml-r2-bar').style.width = `${Math.max(data.regression_r2 * 100, 0)}%`;
        }, 500);

    } catch (e) {
        console.error("Error fetching ML evaluation", e);
    }
}

async function fetchPredictions() {
    try {
        const res = await fetch(`${API_BASE}/models/predictions`);
        const data = await res.json();
        
        const listDiv = document.getElementById('predictive-list');
        listDiv.innerHTML = '';
        
        if (!data.predictions || data.predictions.length === 0) {
            listDiv.innerHTML = '<div style="color: #ef4444; font-style: italic;">Unable to generate forecasts. Check dataset columns.</div>';
            return;
        }
        
        data.predictions.forEach((item, index) => {
            const html = `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 3px solid #8b5cf6;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="color: #94a3b8; font-weight: bold;">#${index+1}</span>
                        <span style="color: #f8f9fa; font-weight: 500;">${item.name}</span>
                    </div>
                    <div style="color: #10b981; font-weight: bold;">
                        +${item.prediction} <span style="font-size: 12px; color: #94a3b8; font-weight: normal;">Units</span>
                    </div>
                </div>
            `;
            listDiv.innerHTML += html;
        });
    } catch (e) {
        console.error("Error fetching predictions", e);
    }
}

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    renderCategorySalesChart();
    renderRiskDistributionChart();
    fetchMLEvaluation();
    fetchPredictions();

    // Setup sidebar interactions
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active nav
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            // Switch view
            const targetId = this.getAttribute('data-target');
            viewSections.forEach(section => {
                if(section.id === targetId) {
                    section.classList.add('active');
                } else {
                    section.classList.remove('active');
                }
            });
        });
    });

    // Setup CSV Upload
    const uploadInput = document.getElementById('csv-upload');
    const uploadBtn = document.querySelector('.upload-btn');
    uploadInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        document.getElementById('current-file-name').innerText = file.name;
        const originalText = uploadBtn.innerHTML;
        const formData = new FormData();
        formData.append("file", file);

        try {
            // Optional: Show loading state on button
            uploadBtn.innerHTML = `Uploading...`;

            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                // Refresh data
                fetchStats();
                renderCategorySalesChart();
                renderRiskDistributionChart();
                fetchMLEvaluation();
                fetchPredictions();
                alert("Dataset uploaded and models retrained successfully!");
            } else {
                alert("Failed to upload dataset.");
            }
            uploadBtn.innerHTML = originalText;
        } catch (err) {
            console.error("Upload error:", err);
            alert("An error occurred during upload.");
        }
    });
    // Setup Web Scraper
    const scrapeBtn = document.getElementById('scrape-btn');
    if (scrapeBtn) {
        scrapeBtn.addEventListener('click', async () => {
            const product = document.getElementById('scrape-product').value;
            const price = parseFloat(document.getElementById('scrape-price').value);
            
            if (!product || isNaN(price)) {
                alert("Please enter both a product name and our current price.");
                return;
            }
            
            const originalText = scrapeBtn.innerHTML;
            scrapeBtn.innerHTML = `Scraping...`;
            
            try {
                const res = await fetch(`${API_BASE}/scrape`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ product_name: product, our_price: price })
                });
                
                const data = await res.json();
                
                document.getElementById('scrape-results').style.display = 'block';
                
                if (res.ok) {
                    document.getElementById('comp-avg-price').innerText = `$${data.competitor_avg.toFixed(2)}`;
                    document.getElementById('comp-count').innerText = data.items_scraped;
                    document.getElementById('comp-prescription').innerText = data.prescription;
                } else {
                    document.getElementById('comp-prescription').innerText = "Failed to scrape data. Please try another product.";
                }
            } catch (err) {
                console.error(err);
                alert("Scraping request failed.");
            }
            
            scrapeBtn.innerHTML = originalText;
        });
    }
});
