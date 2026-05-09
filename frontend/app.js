const API_BASE = 'http://localhost:8000/api';

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

        var chart = new ApexCharts(document.querySelector("#chart-category-sales"), options);
        chart.render();
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

        var chart = new ApexCharts(document.querySelector("#chart-risk-dist"), options);
        chart.render();
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

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    renderCategorySalesChart();
    renderRiskDistributionChart();
    fetchMLEvaluation();

    // Setup sidebar interactions
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
});
