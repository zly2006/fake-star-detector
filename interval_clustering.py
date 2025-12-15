#!/usr/bin/env python3
"""
使用scipy进行时间间隔聚类分析
"""
import requests
from datetime import datetime
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import zscore
from collections import Counter
import json

TOKEN = "ghp_xxx"
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3.star+json"}

print("⏱️  科学聚类分析 - 使用scipy\n")

# 获取前100个star
response = requests.get(
    "https://api.github.com/repos/XiaomingX/indie-hacker-tools-plus/stargazers",
    headers=HEADERS, params={"per_page": 100}
)
stars = response.json()

# 提取时间并计算间隔
times = sorted([datetime.strptime(s['starred_at'], '%Y-%m-%dT%H:%M:%SZ') for s in stars])
intervals = np.array([(times[i] - times[i-1]).total_seconds() for i in range(1, len(times))])

print(f"✅ 分析 {len(intervals)} 个时间间隔\n")

print("="*70)
print("📊 层次聚类分析")
print("="*70)

# 转换为分钟并reshape为2D
intervals_min = intervals / 60
X = intervals_min.reshape(-1, 1)

# 层次聚类
linkage_matrix = linkage(X, method='ward')

# 根据距离阈值切分为簇（尝试分成5-8个簇）
max_clusters = min(8, len(intervals) // 10)
clusters = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')

# 分析每个簇
cluster_info = {}
for cluster_id in range(1, max_clusters + 1):
    cluster_data = intervals_min[clusters == cluster_id]
    if len(cluster_data) > 0:
        cluster_info[cluster_id] = {
            'count': len(cluster_data),
            'mean': float(np.mean(cluster_data)),
            'std': float(np.std(cluster_data)),
            'min': float(np.min(cluster_data)),
            'max': float(np.max(cluster_data))
        }

# 按数量排序
sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['count'], reverse=True)

print(f"\n识别出 {len(cluster_info)} 个簇:\n")
for cluster_id, info in sorted_clusters:
    print(f"簇 {cluster_id}:")
    print(f"   样本数: {info['count']} ({info['count']/len(intervals)*100:.1f}%)")
    print(f"   中心: {info['mean']:.1f}分钟")
    print(f"   范围: {info['min']:.1f} - {info['max']:.1f}分钟")
    print(f"   标准差: {info['std']:.1f}")
    
    # 判断是否是固定间隔
    if info['std'] < 5 and info['count'] >= 5:
        print(f"   🔴 高度规律！疑似自动化")
    elif info['std'] < 10 and info['count'] >= 3:
        print(f"   🟡 较为规律")
    print()

# Z-score异常检测
print("="*70)
print("📈 异常值检测 (Z-score)")
print("="*70)

z_scores = np.abs(zscore(intervals_min))
outliers = np.where(z_scores > 2)[0]

print(f"\n异常值数量: {len(outliers)}/{len(intervals)} ({len(outliers)/len(intervals)*100:.1f}%)")
if len(outliers) > 0:
    print(f"异常间隔(分钟): {[f'{intervals_min[i]:.0f}' for i in outliers[:5]]}")

# 检测周期性（自相关）
print(f"\n{'='*70}")
print("🔄 周期性检测")
print("="*70)

# 检查特定时间点的集中度
star_hours = [t.hour for t in times]
star_minutes = [t.minute for t in times]

hour_counter = Counter(star_hours)
most_common_hours = hour_counter.most_common(5)

print(f"\n最集中的小时:")
for hour, count in most_common_hours:
    print(f"   {hour:02d}:00 - {count}次 ({count/len(times)*100:.1f}%)")
    if count > len(times) * 0.15:
        print(f"      🔴 集中度>15%，疑似定时任务")

# 检查整点分钟分布
minute_ranges = {
    '整点(0-5分)': sum(1 for m in star_minutes if 0 <= m <= 5),
    '半点(25-35分)': sum(1 for m in star_minutes if 25 <= m <= 35),
}

print(f"\n时间点分布:")
for range_name, count in minute_ranges.items():
    pct = count / len(star_minutes) * 100
    print(f"   {range_name}: {count}次 ({pct:.1f}%)")
    if pct > 20:
        print(f"      🔴 高度集中，疑似程序控制")

# 综合判断
print(f"\n{'='*70}")
print("🎯 自动化程度评估")
print("="*70)

score = 0
evidence = []

# 检查主要簇的规律性
if sorted_clusters:
    main_cluster = sorted_clusters[0][1]
    if main_cluster['std'] < 5 and main_cluster['count'] >= 5:
        score += 40
        evidence.append(f"主簇标准差<5分钟，高度规律({main_cluster['count']}个样本)")
    elif main_cluster['std'] < 10:
        score += 20
        evidence.append(f"主簇标准差<10分钟，较为规律")

# 检查整点集中度
if minute_ranges['整点(0-5分)'] / len(star_minutes) > 0.2:
    score += 30
    evidence.append(f"整点附近集中度{minute_ranges['整点(0-5分)']/len(star_minutes)*100:.0f}%")

# 检查小时集中度
if most_common_hours[0][1] > len(times) * 0.15:
    score += 20
    evidence.append(f"{most_common_hours[0][0]}时集中度{most_common_hours[0][1]/len(times)*100:.0f}%")

# 检查簇的数量（如果簇很少说明模式单一）
if len(cluster_info) <= 3:
    score += 10
    evidence.append(f"仅{len(cluster_info)}个主要模式，行为单一")

print(f"\n自动化可疑度: {score}/100\n")

if evidence:
    print("证据:")
    for e in evidence:
        print(f"   • {e}")
    print()

if score >= 70:
    print("🔴 结论: 高度疑似程序自动化刷star")
elif score >= 50:
    print("🟡 结论: 存在明显自动化特征")
else:
    print("🟢 结论: 自动化特征不明显")

# 保存详细结果
result = {
    'clusters': {str(k): v for k, v in cluster_info.items()},
    'automation_score': score,
    'evidence': evidence,
    'outliers_count': len(outliers),
    'total_intervals': len(intervals)
}

with open('clustering_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ 详细结果已保存到 clustering_result.json")
print("="*70)
