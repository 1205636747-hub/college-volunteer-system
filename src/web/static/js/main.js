// 高考志愿推荐系统 - 前端JavaScript

// 全局变量
let currentRecommendations = {};
let comparisonList = [];

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    initEventListeners();
});

// 初始化事件监听
function initEventListeners() {
    // 院校对比功能
    const compareBtn = document.getElementById('compare-btn');
    if (compareBtn) {
        compareBtn.addEventListener('click', showComparison);
    }
}

// 格式化数字
function formatNumber(num) {
    if (num >= 10000) {
        return (num / 10000).toFixed(1) + '万';
    }
    return num.toString();
}

// 显示提示消息
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// 导出推荐结果
function exportResults() {
    if (!currentRecommendations.rush) {
        showToast('暂无推荐结果可导出', 'error');
        return;
    }

    const allResults = [
        ...currentRecommendations.rush.map(r => ({...r, category: '冲'})),
        ...currentRecommendations.stable.map(r => ({...r, category: '稳'})),
        ...currentRecommendations.safe.map(r => ({...r, category: '保'}))
    ];

    // 转换为CSV
    const headers = ['院校', '专业', '平均分', '录取概率', '就业率', '平均薪资', '评级', '推荐类型'];
    const rows = allResults.map(r => [
        r.college, r.major, r.avg_score,
        r.probability + '%', r.employment_rate + '%',
        r.avg_salary, r.rating, r.category
    ]);

    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');

    // 下载
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `志愿推荐结果_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();

    showToast('导出成功', 'success');
}

// 打印推荐结果
function printResults() {
    window.print();
}

// 添加到对比列表
function addToComparison(collegeName) {
    if (comparisonList.includes(collegeName)) {
        comparisonList = comparisonList.filter(c => c !== collegeName);
        showToast(`已从对比列表中移除 ${collegeName}`, 'info');
    } else {
        if (comparisonList.length >= 5) {
            showToast('最多只能对比5所院校', 'error');
            return;
        }
        comparisonList.push(collegeName);
        showToast(`已添加 ${collegeName} 到对比列表`, 'success');
    }

    updateComparisonBar();
}

// 更新对比栏
function updateComparisonBar() {
    const bar = document.getElementById('comparison-bar');
    if (!bar) return;

    if (comparisonList.length === 0) {
        bar.style.display = 'none';
        return;
    }

    bar.style.display = 'flex';
    bar.innerHTML = `
        <div class="comparison-items">
            ${comparisonList.map(c => `
                <span class="comparison-item">
                    ${c}
                    <button onclick="addToComparison('${c}')" class="remove-btn">×</button>
                </span>
            `).join('')}
        </div>
        <button onclick="showComparison()" class="btn btn-primary btn-sm">
            对比 (${comparisonList.length})
        </button>
        <button onclick="clearComparison()" class="btn btn-secondary btn-sm">
            清空
        </button>
    `;
}

// 显示对比结果
async function showComparison() {
    if (comparisonList.length < 2) {
        showToast('请至少选择2所院校进行对比', 'error');
        return;
    }

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ colleges: comparisonList })
        });

        const data = await response.json();

        if (data.success) {
            renderComparisonModal(data.comparison);
        }
    } catch (error) {
        showToast('对比失败: ' + error.message, 'error');
    }
}

// 渲染对比弹窗
function renderComparisonModal(comparison) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>院校对比</h3>
                <button class="close-btn" onclick="this.closest('.modal').remove()">&times;</button>
            </div>
            <div class="modal-body">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>指标</th>
                            ${Object.keys(comparison).map(c => `<th>${c}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>平均录取分</td>
                            ${Object.values(comparison).map(v => `<td>${v.avg_score}</td>`).join('')}
                        </tr>
                        <tr>
                            <td>就业率</td>
                            ${Object.values(comparison).map(v => `<td>${v.employment_rate}%</td>`).join('')}
                        </tr>
                        <tr>
                            <td>平均薪资</td>
                            ${Object.values(comparison).map(v => `<td>${formatNumber(v.avg_salary)}</td>`).join('')}
                        </tr>
                        <tr>
                            <td>学科评级</td>
                            ${Object.values(comparison).map(v => `<td>${v.rating}</td>`).join('')}
                        </tr>
                        <tr>
                            <td>专业数量</td>
                            ${Object.values(comparison).map(v => `<td>${v.major_count}</td>`).join('')}
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

// 清空对比列表
function clearComparison() {
    comparisonList = [];
    updateComparisonBar();
    showToast('已清空对比列表', 'info');
}

// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .modal-content {
        background: white;
        border-radius: 15px;
        max-width: 800px;
        width: 90%;
        max-height: 80vh;
        overflow: auto;
    }
    
    .modal-header {
        padding: 20px;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .modal-body {
        padding: 20px;
    }
    
    .close-btn {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #999;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .comparison-table th {
        background: #f8f9fa;
        font-weight: 600;
    }
    
    #comparison-bar {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: white;
        padding: 15px 25px;
        border-radius: 50px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 20px;
        z-index: 999;
    }
    
    .comparison-items {
        display: flex;
        gap: 10px;
    }
    
    .comparison-item {
        background: #667eea;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .remove-btn {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        font-size: 1.2rem;
    }
    
    .btn-sm {
        padding: 8px 16px;
        font-size: 0.9rem;
    }
    
    @media print {
        .navbar, .input-panel, .result-tabs, .footer, .hero-buttons {
            display: none !important;
        }
        
        .result-panel {
            box-shadow: none;
        }
    }
`;
document.head.appendChild(style);