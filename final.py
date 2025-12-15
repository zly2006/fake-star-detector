#!/usr/bin/env python3
"""
Comprehensive Star Manipulation Detection Tool
Usage: python3 final.py <owner> <repo>
"""
import sys
import os
import re
import requests
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('GITHUB_TOKEN')
if not TOKEN:
    print("Error: GITHUB_TOKEN not found in .env file")
    sys.exit(1)

HEADERS = {"Authorization": f"token {TOKEN}"}
STAR_HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3.star+json"}

def get_total_count_from_search(owner, repo, item_type):
    """Get accurate total count using GitHub Search API"""
    query = f"repo:{owner}/{repo} type:{item_type}"
    url = "https://api.github.com/search/issues"
    
    try:
        r = requests.get(url, headers=HEADERS, params={"q": query, "per_page": 1})
        if r.status_code == 200:
            return r.json().get('total_count', 0)
        else:
            return 0
    except:
        return 0

def create_visualization(owner, repo, report_data, stargazers_data, intervals_min, times, clusters, max_clusters):
    """Create 4-panel visualization"""
    print(f"\n[7/8] Creating visualization...")
    
    metrics = report_data['metrics']
    main_cluster = metrics.get('main_cluster', {})
    
    # Calculate half-hour peak
    star_minutes = [t.minute for t in times]
    near_half = sum(1 for m in star_minutes if 25 <= m <= 35)
    half_hour_pct = near_half / len(times) * 100
    
    # Filter data for plot 1: only show intervals < 500 minutes
    intervals_filtered = intervals_min[intervals_min < 500]
    in_range_count = len(intervals_filtered)
    total_count = len(intervals_min)
    in_range_pct = in_range_count / total_count * 100
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Star Manipulation Evidence - {owner}/{repo}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Interval Distribution (< 500 min only)
    ax1 = axes[0, 0]
    ax1.hist(intervals_filtered, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(main_cluster['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Main cluster: {main_cluster['mean']:.1f} min")
    ax1.set_xlabel('Time Interval (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Star Time Interval Distribution (<500 min)\n{in_range_count}/{total_count} stars ({in_range_pct:.1f}%) in this range')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster Visualization
    ax2 = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, max_clusters))
    for cluster_id in range(1, max_clusters + 1):
        cluster_data = intervals_min[clusters == cluster_id]
        if len(cluster_data) > 0:
            ax2.scatter([cluster_id] * len(cluster_data), cluster_data, 
                       c=[colors[cluster_id-1]], alpha=0.6, s=50)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Interval (minutes)')
    ax2.set_title('Hierarchical Clustering Results')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time of Day Distribution
    ax3 = axes[1, 0]
    hours = [t.hour for t in times]
    hour_counts = Counter(hours)
    ax3.bar(hour_counts.keys(), hour_counts.values(), color='coral', edgecolor='black')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Number of Stars')
    ax3.set_title('Star Distribution by Hour')
    ax3.set_xticks(range(24))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Key Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Determine status indicators
    issue_status = "Suspicious: < 1%" if metrics['issue_rate'] < 1 else "OK"
    pr_status = "Suspicious: < 1%" if metrics['pr_rate'] < 1 else "OK"
    fork_status = "Suspicious: < 8%" if metrics['fork_rate'] < 8 else "OK"
    bot_status = "Suspicious: > 80%" if metrics['bot_commit_ratio'] > 80 else "OK"
    
    metrics_text = f"""
KEY EVIDENCE SUMMARY

Repository Stats:
• Total Stars: {metrics['stars']}
• Issue Rate: {metrics['issue_rate']:.2f}% ({issue_status})
• PR Rate: {metrics['pr_rate']:.2f}% ({pr_status})
• Fork Rate: {metrics['fork_rate']:.1f}% ({fork_status})
• Bot Commits: {metrics['bot_commit_ratio']:.0f}% ({bot_status})

Time Pattern Analysis:
• Main Cluster: {main_cluster['percentage']:.1f}%
• Mean Interval: {main_cluster['mean']:.1f} min
• Std Deviation: {main_cluster['std']:.1f} min
• Half-hour Peak: {half_hour_pct:.0f}%

Suspicion Score: {report_data['suspicion_score']}/{report_data['max_score']}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_file = f"visualization_{owner}_{repo}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()

def generate_verdict(owner, repo, report_data):
    """Generate detailed verdict markdown file"""
    print(f"\n[8/8] Generating verdict document...")
    
    metrics = report_data['metrics']
    evidence = report_data['evidence_scores']
    total_score = report_data['suspicion_score']
    
    # Determine verdict level
    if total_score >= 100:
        verdict_level = "🔴 CONFIRMED MANIPULATION"
        confidence = "极高"
    elif total_score >= 60:
        verdict_level = "🔴 HIGH SUSPICION"
        confidence = "高"
    elif total_score >= 30:
        verdict_level = "🟡 MEDIUM SUSPICION"
        confidence = "中"
    else:
        verdict_level = "🟢 LOW SUSPICION"
        confidence = "低"
    
    verdict_md = f"""# 分析报告 - {owner}/{repo}

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 最终判决

### 可疑度评分: **{total_score}/{report_data['max_score']}**

### 判定结果: **{verdict_level}**

### 置信度: **{confidence}**

---

## 📊 基础数据

| 指标 | 数值 | 状态 |
|------|------|------|
| Stars | {metrics['stars']} | - |
| Forks | {metrics['forks']} ({metrics['fork_rate']:.1f}%) | {'🔴' if metrics['fork_rate'] < 8 else '🟢'} |
| Issues | {metrics['total_issues']} ({metrics['issue_rate']:.2f}%) | {'🔴' if metrics['issue_rate'] < 2 else '🟢'} |
| PRs | {metrics['total_prs']} ({metrics['pr_rate']:.2f}%) | {'🔴' if metrics['pr_rate'] < 2 else '🟢'} |
| Bot Commits | {metrics['bot_commit_ratio']:.0f}% | {'🔴' if metrics['bot_commit_ratio'] > 50 else '🟢'} |

---

## 🔍 证据详情

### 1. Issue率分析 ({evidence['issue_rate']} 分)

- **实际值**: {metrics['issue_rate']:.2f}%
- **正常值**: >2%
- **判定**: {'🔴 异常 - Issue率过低' if evidence['issue_rate'] > 0 else '🟢 正常'}

{'**说明**: Issue率<1%说明用户只收藏不使用，典型的虚假star特征。' if evidence['issue_rate'] >= 30 else '**说明**: Issue率正常，用户有真实反馈。' if evidence['issue_rate'] == 0 else '**说明**: Issue率略低，需关注。'}

### 2. PR率分析 ({evidence['pr_rate']} 分)

- **实际值**: {metrics['pr_rate']:.2f}%
- **正常值**: >2%
- **判定**: {'🔴 异常 - PR率过低' if evidence['pr_rate'] > 0 else '🟢 正常'}

{'**说明**: 几乎无PR说明项目无人贡献，缺乏真实用户参与。' if evidence['pr_rate'] >= 20 else '**说明**: PR率正常，项目有贡献者。' if evidence['pr_rate'] == 0 else '**说明**: PR率略低。'}

### 3. Fork率分析 ({evidence['fork_rate']} 分)

- **实际值**: {metrics['fork_rate']:.1f}%
- **正常值**: >8%
- **判定**: {'🔴 异常 - Fork率过低' if evidence['fork_rate'] > 0 else '🟢 正常'}

{'**说明**: Fork率<8%说明用户不实际使用项目，只是收藏。' if evidence['fork_rate'] > 0 else '**说明**: Fork率正常，用户真实使用项目。'}

### 4. Bot提交分析 ({evidence['bot_commits']} 分)

- **实际值**: {metrics['bot_commit_ratio']:.0f}%
- **正常值**: <20%
- **判定**: {'🔴 严重异常 - Bot刷活跃度' if evidence['bot_commits'] >= 30 else '🟡 轻度异常' if evidence['bot_commits'] > 0 else '🟢 正常'}

{'**说明**: Bot提交占比>80%，明显用于刷活跃度和trending排名。' if evidence['bot_commits'] >= 30 else '**说明**: 无Bot提交，提交记录真实。' if evidence['bot_commits'] == 0 else '**说明**: 少量Bot提交。'}

### 5. 时间聚类分析 ({evidence['time_clustering']} 分) ⭐ 核心证据

"""

    if 'main_cluster' in metrics and metrics['main_cluster']:
        cluster = metrics['main_cluster']
        verdict_md += f"""
- **主簇大小**: {cluster['count']} 样本 ({cluster['percentage']:.1f}%)
- **平均间隔**: {cluster['mean']:.1f} 分钟
- **标准差**: {cluster['std']:.1f} 分钟
- **判定**: {'🔴 极度异常 - 程序自动化' if evidence['time_clustering'] >= 50 else '🟡 轻度异常' if evidence['time_clustering'] > 0 else '🟢 正常'}

{'**关键发现**: 标准差<5分钟，' + str(int(cluster['percentage'])) + '%的star高度集中！这在统计学上不可能是人类行为，明确指向程序自动化控制。' if evidence['time_clustering'] >= 50 else '**说明**: 时间分布正常，符合人类行为模式。' if evidence['time_clustering'] == 0 else '**说明**: 存在一定规律性。'}
"""
    else:
        verdict_md += "\n数据不足，无法进行聚类分析。\n"

    verdict_md += f"""
### 6. 批量创建分析 ({evidence['bulk_creation']} 分)

- **判定**: {'🔴 异常 - 发现批量创建' if evidence['bulk_creation'] > 0 else '🟢 正常'}

---

## 🎯 最终结论

"""

    if total_score >= 100:
        verdict_md += f"""
### ⚠️  确认存在Star操纵行为

基于多维度证据分析，该仓库存在**明确的Star操纵行为**。

#### 建议:
- 可向GitHub Support举报
- 提供本分析报告作为证据
"""
    elif total_score >= 60:
        verdict_md += "### ⚠️  高度可疑\n\n该仓库存在多个异常指标。\n"
    elif total_score >= 30:
        verdict_md += "### ⚠️  中度可疑\n\n存在部分异常指标。\n"
    else:
        verdict_md += "### ✅ 正常项目\n\n各项指标均在正常范围内。\n"

    verdict_md += f"""
---

**生成工具**: https://github.com/zly2006/fake-star-detector# v2.0  
**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**报告格式**: Markdown

---

## 📎 附件

- 详细数据: `report_{owner}_{repo}.json`
- 可视化图表: ![visualization](visualization_{owner}_{repo}.png)
"""

    output_file = f"verdict_{owner}_{repo}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(verdict_md)
    
    print(f"   ✓ Saved: {output_file}")

def analyze_repository(owner, repo):
    """Main analysis function"""
    
    print("="*70)
    print("🔍 COMPREHENSIVE STAR MANIPULATION DETECTION")
    print("="*70)
    print(f"\nTarget: {owner}/{repo}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get repository data
    print("[1/6] Fetching repository data...")
    repo_r = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=HEADERS)
    if repo_r.status_code != 200:
        print(f"❌ Error: Repository not found ({repo_r.status_code})")
        sys.exit(1)
    
    repo_data = repo_r.json()
    stars = repo_data['stargazers_count']
    forks = repo_data['forks_count']
    
    # Get counts via Search API
    print("[2/6] Fetching issues and PRs...")
    total_issues = get_total_count_from_search(owner, repo, 'issue')
    total_prs = get_total_count_from_search(owner, repo, 'pr')
    
    issue_rate = total_issues / stars * 100 if stars > 0 else 0
    pr_rate = total_prs / stars * 100 if stars > 0 else 0
    fork_rate = forks / stars * 100 if stars > 0 else 0
    
    print(f"   ✓ Stars: {stars}")
    print(f"   ✓ Forks: {forks} ({fork_rate:.1f}%)")
    print(f"   ✓ Issues: {total_issues} ({issue_rate:.2f}%)")
    print(f"   ✓ PRs: {total_prs} ({pr_rate:.2f}%)")
    
    # Evidence scoring
    evidence_1_score = 30 if issue_rate < 1 and stars > 100 else (15 if issue_rate < 2 and stars > 100 else 0)
    evidence_2_score = 20 if pr_rate < 1 and stars > 100 else (10 if pr_rate < 2 and stars > 100 else 0)
    evidence_3_score = 25 if fork_rate < 8 and stars > 100 else 0
    
    if evidence_1_score >= 30: print(f"   🔴 ANOMALY: Issue rate < 1%")
    if evidence_2_score >= 20: print(f"   🔴 ANOMALY: PR rate < 1%")
    if evidence_3_score > 0: print(f"   🔴 ANOMALY: Fork rate < 8%")
    
    # Commits
    print(f"\n[3/6] Analyzing commits...")
    commits_r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/commits",
                            headers=HEADERS, params={"per_page": 100})
    commits = commits_r.json() if commits_r.status_code == 200 else []
    bot_commits = sum(1 for c in commits if 'Update TIME.md' in c.get('commit', {}).get('message', ''))
    bot_ratio = bot_commits / len(commits) * 100 if commits else 0
    
    print(f"   ✓ Commits: {len(commits)}, Bot: {bot_commits} ({bot_ratio:.0f}%)")
    
    evidence_4_score = 30 if bot_ratio > 80 and len(commits) > 50 else (15 if bot_ratio > 50 else 0)
    if evidence_4_score >= 30: print(f"   🔴 ANOMALY: Bot commits > 80%")
    
    # Clustering
    print(f"\n[4/6] Time clustering analysis...")
    stargazers_r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/stargazers",
                                headers=STAR_HEADERS, params={"per_page": 100})
    stargazers = stargazers_r.json() if stargazers_r.status_code == 200 else []
    
    evidence_5_score = 0
    main_cluster_info = {}
    intervals_min = None
    times = None
    clusters = None
    max_clusters = 0
    
    if len(stargazers) >= 20:
        times = sorted([datetime.strptime(s['starred_at'], '%Y-%m-%dT%H:%M:%SZ') for s in stargazers])
        intervals = np.array([(times[i] - times[i-1]).total_seconds() for i in range(1, len(times))])
        intervals_min = intervals / 60
        
        X = intervals_min.reshape(-1, 1)
        linkage_matrix = linkage(X, method='ward')
        max_clusters = min(8, len(intervals) // 10)
        clusters = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')
        
        cluster_info = {}
        for cid in range(1, max_clusters + 1):
            cluster_data = intervals_min[clusters == cid]
            if len(cluster_data) > 0:
                cluster_info[cid] = {
                    'count': len(cluster_data),
                    'mean': float(np.mean(cluster_data)),
                    'std': float(np.std(cluster_data)),
                    'percentage': len(cluster_data) / len(intervals) * 100
                }
        
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['count'], reverse=True)
        main_cluster_info = sorted_clusters[0][1]
        
        print(f"   ✓ Main cluster: {main_cluster_info['count']} samples, std={main_cluster_info['std']:.1f}min")
        
        if main_cluster_info['std'] < 5 and main_cluster_info['count'] >= 10:
            evidence_5_score = 50
            print(f"   🔴 CRITICAL: Automated pattern!")
        elif main_cluster_info['std'] < 10 and main_cluster_info['percentage'] > 30:
            evidence_5_score = 25
    else:
        print(f"   ⚠️  Insufficient data")
    
    # Bulk creation
    print(f"\n[5/6] Checking patterns...")
    user_repos_r = requests.get(f"https://api.github.com/users/{owner}/repos",
                                headers=HEADERS, params={"per_page": 100})
    all_repos = user_repos_r.json() if user_repos_r.status_code == 200 else []
    
    high_star_repos = [r for r in all_repos if r['stargazers_count'] > 50]
    created_dates = defaultdict(list)
    for r in high_star_repos:
        created_dates[r['created_at'][:10]].append(r['stargazers_count'])
    
    bulk_dates = {d: sum(s) for d, s in created_dates.items() if len(s) >= 2}
    evidence_6_score = 25 if any(len(s) >= 3 for s in created_dates.values()) else (10 if bulk_dates else 0)
    
    if evidence_6_score > 0:
        print(f"   ✓ Found {len(bulk_dates)} bulk dates")
    
    # Total
    total_score = sum([evidence_1_score, evidence_2_score, evidence_3_score, 
                      evidence_4_score, evidence_5_score, evidence_6_score])
    
    print(f"\n[6/6] Saving report...")
    
    status = "🔴 HIGH SUSPICION" if total_score >= 80 else \
             "🟡 MEDIUM SUSPICION" if total_score >= 40 else \
             "🟢 LOW SUSPICION"
    
    report = {
        'analysis_date': datetime.now().isoformat(),
        'repository': f"{owner}/{repo}",
        'metrics': {
            'stars': stars, 'forks': forks, 'fork_rate': fork_rate,
            'total_issues': total_issues, 'issue_rate': issue_rate,
            'total_prs': total_prs, 'pr_rate': pr_rate,
            'bot_commit_ratio': bot_ratio,
            'main_cluster': main_cluster_info
        },
        'suspicion_score': total_score,
        'max_score': 180,
        'status': status,
        'evidence_scores': {
            'issue_rate': evidence_1_score,
            'pr_rate': evidence_2_score,
            'fork_rate': evidence_3_score,
            'bot_commits': evidence_4_score,
            'time_clustering': evidence_5_score,
            'bulk_creation': evidence_6_score
        }
    }
    
    output_file = f"report_{owner}_{repo}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✓ Saved: {output_file}")
    
    # Visualization
    if intervals_min is not None and times is not None:
        create_visualization(owner, repo, report, stargazers, intervals_min, times, clusters, max_clusters)
    
    # Verdict
    generate_verdict(owner, repo, report)
    
    print(f"\n{'='*70}")
    print(f"📊 FINAL SCORE: {total_score}/180")
    print(f"STATUS: {status}")
    print('='*70)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 final.py <owner> <repo>")
        sys.exit(1)
    
    analyze_repository(sys.argv[1], sys.argv[2])
