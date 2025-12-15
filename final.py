#!/usr/bin/env python3
"""
Comprehensive Star Manipulation Detection Tool
Usage: python3 final.py <owner> <repo>
Example: python3 final.py XiaomingX indie-hacker-tools-plus
"""
import sys
import os
import re
import requests
import time
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import zscore
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv('GITHUB_TOKEN')
if not TOKEN:
    print("Error: GITHUB_TOKEN not found in .env file")
    sys.exit(1)

HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
STAR_HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3.star+json"}

def get_total_count_from_api(url, params=None):
    """Get total count using Link header pagination"""
    params = params or {}
    r = requests.get(url, headers=HEADERS, params={**params, "per_page": 1})
    if r.status_code != 200:
        return 0
    
    if 'Link' in r.headers:
        match = re.search(r'page=(\d+)>; rel="last"', r.headers['Link'])
        if match:
            return int(match.group(1))
    
    return len(r.json())

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
        print(f"❌ Error: Repository not found or API error ({repo_r.status_code})")
        sys.exit(1)
    
    repo_data = repo_r.json()
    stars = repo_data['stargazers_count']
    forks = repo_data['forks_count']
    
    # KEY FIX: Get total issues and PRs separately (not just open)
    print("[2/6] Fetching issues and PRs...")
    total_issues = get_total_count_from_api(
        f"https://api.github.com/repos/{owner}/{repo}/issues",
        {"state": "all", "is": "issue"}
    )
    total_prs = get_total_count_from_api(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        {"state": "all"}
    )
    
    # Calculate interaction rate (issues + PRs)
    interaction_rate = (total_issues + total_prs) / stars * 100 if stars > 0 else 0
    fork_rate = forks / stars * 100 if stars > 0 else 0
    
    print(f"   ✓ Stars: {stars}")
    print(f"   ✓ Forks: {forks} ({fork_rate:.1f}%)")
    print(f"   ✓ Total Issues: {total_issues}")
    print(f"   ✓ Total PRs: {total_prs}")
    print(f"   ✓ Interaction Rate: {interaction_rate:.2f}%")
    
    # Evidence 1: Low Interaction Rate (FIXED METRIC)
    evidence_1_score = 0
    if interaction_rate < 3 and stars > 100:
        evidence_1_score = 35
        print(f"   🔴 ANOMALY: Interaction rate < 3% (normal: >3%)")
    elif interaction_rate < 5:
        evidence_1_score = 15
        print(f"   🟡 WARNING: Interaction rate < 5%")
    
    # Evidence 2: Low Fork Rate
    evidence_2_score = 0
    if fork_rate < 8 and stars > 100:
        evidence_2_score = 25
        print(f"   🔴 ANOMALY: Fork rate < 8%")
    
    # Check bot commits
    print(f"\n[3/6] Analyzing commits...")
    commits_r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits",
        headers=HEADERS, params={"per_page": 100}
    )
    commits = commits_r.json() if commits_r.status_code == 200 else []
    bot_commits = sum(1 for c in commits 
                     if 'Update TIME.md' in c.get('commit', {}).get('message', ''))
    bot_ratio = bot_commits / len(commits) * 100 if commits else 0
    
    print(f"   ✓ Total Commits (sample): {len(commits)}")
    print(f"   ✓ Bot Commits: {bot_commits} ({bot_ratio:.0f}%)")
    
    # Evidence 3: Bot Commits
    evidence_3_score = 0
    if bot_ratio > 80 and len(commits) > 50:
        evidence_3_score = 30
        print(f"   🔴 ANOMALY: Bot commits > 80%")
    elif bot_ratio > 50:
        evidence_3_score = 15
        print(f"   🟡 WARNING: Bot commits > 50%")
    
    # Time interval clustering
    print(f"\n[4/6] Fetching stargazers for clustering...")
    stargazers_r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/stargazers",
        headers=STAR_HEADERS, params={"per_page": 100}
    )
    stargazers = stargazers_r.json() if stargazers_r.status_code == 200 else []
    
    evidence_5_score = 0
    main_cluster_info = {}
    
    if len(stargazers) >= 20:
        print(f"   ✓ Analyzing {len(stargazers)} early stars...")
        
        times = sorted([datetime.strptime(s['starred_at'], '%Y-%m-%dT%H:%M:%SZ') 
                       for s in stargazers])
        intervals = np.array([(times[i] - times[i-1]).total_seconds() 
                             for i in range(1, len(times))])
        
        intervals_min = intervals / 60
        X = intervals_min.reshape(-1, 1)
        
        # Hierarchical clustering
        linkage_matrix = linkage(X, method='ward')
        max_clusters = min(8, len(intervals) // 10)
        clusters = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')
        
        # Analyze clusters
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
        
        sorted_clusters = sorted(cluster_info.items(), 
                                key=lambda x: x[1]['count'], reverse=True)
        main_cluster_info = sorted_clusters[0][1]
        
        print(f"\n   Main Cluster:")
        print(f"   • Size: {main_cluster_info['count']} ({main_cluster_info['percentage']:.1f}%)")
        print(f"   • Mean: {main_cluster_info['mean']:.1f} min")
        print(f"   • Std: {main_cluster_info['std']:.1f} min")
        
        # Evidence 5: Time Clustering
        if main_cluster_info['std'] < 5 and main_cluster_info['count'] >= 10:
            evidence_5_score = 50
            print(f"   🔴 CRITICAL: Std < 5 min with {main_cluster_info['count']} samples!")
        elif main_cluster_info['std'] < 10 and main_cluster_info['percentage'] > 30:
            evidence_5_score = 25
            print(f"   🟡 WARNING: Regular pattern detected")
    else:
        print(f"   ⚠️  Not enough stargazers for clustering analysis")
    
    # Check user's other repos for bulk creation
    print(f"\n[5/6] Checking user's repository pattern...")
    user_repos_r = requests.get(
        f"https://api.github.com/users/{owner}/repos",
        headers=HEADERS, params={"per_page": 100, "sort": "stargazers_count"}
    )
    all_repos = user_repos_r.json() if user_repos_r.status_code == 200 else []
    
    high_star_repos = [r for r in all_repos if r['stargazers_count'] > 50]
    created_dates = defaultdict(list)
    
    for r in high_star_repos:
        date = r['created_at'][:10]
        created_dates[date].append(r['stargazers_count'])
    
    bulk_dates = {d: sum(stars_list) for d, stars_list in created_dates.items() 
                  if len(stars_list) >= 2}
    
    evidence_4_score = 0
    if bulk_dates:
        print(f"   ✓ Found {len(bulk_dates)} dates with multiple high-star repos")
        for date, total in list(bulk_dates.items())[:2]:
            print(f"      {date}: {total} stars")
        
        if any(len(stars_list) >= 3 for stars_list in created_dates.values()):
            evidence_4_score = 25
            print(f"   🔴 ANOMALY: 3+ repos created on same day")
        elif bulk_dates:
            evidence_4_score = 10
    
    # Calculate total score
    total_score = (evidence_1_score + evidence_2_score + evidence_3_score + 
                   evidence_4_score + evidence_5_score)
    
    print(f"\n[6/6] Generating report...")
    print(f"\n{'='*70}")
    print(f"📊 FINAL SCORE: {total_score}/165")
    print('='*70)
    
    print(f"\nEvidence Breakdown:")
    print(f"   1. Interaction Rate: {evidence_1_score} points")
    print(f"   2. Fork Rate: {evidence_2_score} points")
    print(f"   3. Bot Commits: {evidence_3_score} points")
    print(f"   4. Bulk Creation: {evidence_4_score} points")
    print(f"   5. Time Clustering: {evidence_5_score} points")
    
    status = "🔴 HIGH SUSPICION" if total_score >= 80 else \
             "🟡 MEDIUM SUSPICION" if total_score >= 40 else \
             "🟢 LOW SUSPICION"
    
    print(f"\nStatus: {status}")
    
    # Save report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'repository': f"{owner}/{repo}",
        'metrics': {
            'stars': stars,
            'forks': forks,
            'fork_rate': fork_rate,
            'total_issues': total_issues,
            'total_prs': total_prs,
            'interaction_rate': interaction_rate,
            'bot_commit_ratio': bot_ratio
        },
        'suspicion_score': total_score,
        'status': status,
        'evidence_scores': {
            'interaction_rate': evidence_1_score,
            'fork_rate': evidence_2_score,
            'bot_commits': evidence_3_score,
            'bulk_creation': evidence_4_score,
            'time_clustering': evidence_5_score
        }
    }
    
    output_file = f"report_{owner}_{repo}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {output_file}")
    print('='*70)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 final.py <owner> <repo>")
        print("Example: python3 final.py XiaomingX indie-hacker-tools-plus")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    
    analyze_repository(owner, repo)
