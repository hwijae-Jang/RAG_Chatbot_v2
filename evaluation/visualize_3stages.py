
'''
3단계 LLM 평가 가중치 조절 시각화

'''
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_results(filename: str) -> dict:
    """평가 결과 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_3stage_keyword_accuracy(before_m, middle_m, after_m):
    """3단계 키워드 정확도 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))

    versions = ['Before', 'Middle', 'After']
    scores = [
        before_m['avg_keyword_score'] * 100,
        middle_m['avg_keyword_score'] * 100,
        after_m['avg_keyword_score'] * 100
    ]

    colors = ['#ff6b6b', '#ffd93d', '#51cf66']
    bars = ax.bar(versions, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    # 값 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 개선율 화살표 (Before → Middle)
    improvement1 = scores[1] - scores[0]
    ax.annotate(f'+{improvement1:.1f}%p\n(Document\nOptimization)',
                xy=(1, scores[1]), xytext=(0.5, (scores[0] + scores[1])/2 + 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                fontsize=11, color='orange', fontweight='bold', ha='center')

    # 개선율 화살표 (Middle → After)
    improvement2 = scores[2] - scores[1]
    ax.annotate(f'+{improvement2:.1f}%p\n(Code\nOptimization)',
                xy=(2, scores[2]), xytext=(1.5, (scores[1] + scores[2])/2 + 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, color='green', fontweight='bold', ha='center')

    ax.set_ylabel('Keyword Accuracy (%)', fontsize=13)
    ax.set_title('3-Stage Improvement: Keyword Accuracy', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # 목표선
    ax.axhline(y=70, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 70%')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('3stage_keyword_accuracy.png', dpi=300, bbox_inches='tight')
    print("📊 저장: 3stage_keyword_accuracy.png")


def plot_3stage_search_success(before_m, middle_m, after_m):
    """3단계 검색 성공률 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))

    versions = ['Before', 'Middle', 'After']
    rates = [
        before_m['search_success_rate'] * 100,
        middle_m['search_success_rate'] * 100,
        after_m['search_success_rate'] * 100
    ]

    colors = ['#ff6b6b', '#ffd93d', '#51cf66']
    bars = ax.bar(versions, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    # 값 표시
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 개선율 표시
    improvement1 = rates[1] - rates[0]
    if improvement1 > 0:
        ax.annotate(f'+{improvement1:.1f}%p',
                    xy=(1, rates[1]), xytext=(0.5, (rates[0] + rates[1])/2 + 2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                    fontsize=11, color='orange', fontweight='bold')

    improvement2 = rates[2] - rates[1]
    if improvement2 > 0:
        ax.annotate(f'+{improvement2:.1f}%p',
                    xy=(2, rates[2]), xytext=(1.5, (rates[1] + rates[2])/2 + 2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=11, color='green', fontweight='bold')

    ax.set_ylabel('Search Success Rate (%)', fontsize=13)
    ax.set_title('3-Stage Improvement: Search Success Rate', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # 목표선
    ax.axhline(y=100, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 100%')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('3stage_search_success.png', dpi=300, bbox_inches='tight')
    print("📊 저장: 3stage_search_success.png")


def plot_improvement_breakdown(before_m, middle_m, after_m):
    """개선 기여도 분해 분석"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 키워드 정확도 개선 분해
    before_kw = before_m['avg_keyword_score'] * 100
    middle_kw = middle_m['avg_keyword_score'] * 100
    after_kw = after_m['avg_keyword_score'] * 100

    doc_improvement = middle_kw - before_kw
    code_improvement = after_kw - middle_kw
    total_improvement = after_kw - before_kw

    # For the pie chart, we only consider positive contributions.
    # If a stage resulted in a decrease, its 'contribution to improvement' for the pie chart is 0.
    positive_doc_improvement_for_pie = max(0, doc_improvement)
    positive_code_improvement_for_pie = max(0, code_improvement)

    # Sum of only the positive improvements for the pie chart's denominator
    sum_of_positive_improvements = positive_doc_improvement_for_pie + positive_code_improvement_for_pie

    pie_labels = []
    pie_sizes_for_chart = []
    pie_colors_for_chart = []
    colors_map = {'Document': '#ffd93d', 'Code': '#51cf66'}

    if sum_of_positive_improvements > 0:
        if positive_doc_improvement_for_pie > 0:
            pie_labels.append(f'Document\nOptimization\n({doc_improvement:.1f}%p)')
            pie_sizes_for_chart.append(positive_doc_improvement_for_pie)
            pie_colors_for_chart.append(colors_map['Document'])

        if positive_code_improvement_for_pie > 0:
            pie_labels.append(f'Code\nOptimization\n({code_improvement:.1f}%p)')
            pie_sizes_for_chart.append(positive_code_improvement_for_pie)
            pie_colors_for_chart.append(colors_map['Code'])

    # 파이 차트
    if not pie_sizes_for_chart:
        ax1.text(0, 0, 'No Positive\nContribution to Improvement', horizontalalignment='center', verticalalignment='center',
                 fontsize=12, color='gray', fontweight='bold')
    else:
        wedges, texts = ax1.pie(pie_sizes_for_chart, labels=pie_labels, colors=pie_colors_for_chart, startangle=90,
                                 textprops={'fontsize': 11, 'fontweight': 'bold'}, wedgeprops=dict(width=0.3))

    ax1.set_title('Improvement Contribution\n(Keyword Accuracy)', fontsize=13, fontweight='bold')

    # 누적 개선 그래프
    stages = ['Before', 'Middle', 'After']

    ax2.plot(stages, [before_kw, middle_kw, after_kw], marker='o', markersize=12,
             linewidth=3, color='#4dabf7', label='Actual Score')
    ax2.fill_between(range(3), before_kw, [before_kw, middle_kw, after_kw],
                     alpha=0.3, color='#4dabf7')

    # 개선량 표시
    ax2.annotate(f'+{doc_improvement:.1f}%p\nDoc Opt',
                xy=(1, middle_kw), xytext=(0.7, middle_kw + 5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                fontsize=10, color='orange', fontweight='bold')

    ax2.annotate(f'{code_improvement:+.1f}%p\nCode Opt',
                xy=(2, after_kw), xytext=(1.7, after_kw + 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, color='green', fontweight='bold')

    ax2.set_ylabel('Keyword Accuracy (%)', fontsize=12)
    ax2.set_title('Cumulative Improvement', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('improvement_breakdown.png', dpi=300, bbox_inches='tight')
    print("📊 저장: improvement_breakdown.png")


def plot_response_time_comparison(before_m, middle_m, after_m):
    """3단계 응답 시간 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))

    versions = ['Before', 'Middle', 'After']
    times = [
        before_m['avg_response_time'],
        middle_m['avg_response_time'],
        after_m['avg_response_time']
    ]

    colors = ['#ff6b6b', '#ffd93d', '#51cf66']
    bars = ax.bar(versions, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    # 값 표시
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel('Average Response Time (seconds)', fontsize=13)
    ax.set_title('3-Stage Comparison: Response Time', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 목표선
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Limit: 3.0s')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('3stage_response_time.png', dpi=300, bbox_inches='tight')
    print("📊 저장: 3stage_response_time.png")


def generate_3stage_report(data):
    """3단계 텍스트 리포트"""
    before_m = data['before']['metrics']
    middle_m = data['middle']['metrics']
    after_m = data['after']['metrics']

    # 개선율 계산
    kw_b_m = (middle_m['avg_keyword_score'] - before_m['avg_keyword_score']) * 100
    kw_m_a = (after_m['avg_keyword_score'] - middle_m['avg_keyword_score']) * 100
    kw_total = (after_m['avg_keyword_score'] - before_m['avg_keyword_score']) * 100

    sr_b_m = (middle_m['search_success_rate'] - before_m['search_success_rate']) * 100
    sr_m_a = (after_m['search_success_rate'] - middle_m['search_success_rate']) * 100
    sr_total = (after_m['search_success_rate'] - before_m['search_success_rate']) * 100

    # 기여율 계산
    doc_contribution = (kw_b_m / kw_total * 100) if kw_total > 0 else 0
    code_contribution = (kw_m_a / kw_total * 100) if kw_total > 0 else 0

    report = f"""
{'='*70}
항공권 환불 챗봇 - 3단계 성능 평가 리포트
{'='*70}

📅 평가 일시: {data['timestamp']}
📊 총 평가 질문: 15개

{'='*70}
단계별 성능 비교
{'='*70}

1. 키워드 정확도
   Before: {before_m['avg_keyword_score']:.1%}
   Middle: {middle_m['avg_keyword_score']:.1%} (+{kw_b_m:.1f}%p)
   After:  {after_m['avg_keyword_score']:.1%} (+{kw_m_a:.1f}%p)
   총 개선: +{kw_total:.1f}%p

2. 검색 성공률
   Before: {before_m['search_success_rate']:.1%}
   Middle: {middle_m['search_success_rate']:.1%} (+{sr_b_m:.1f}%p)
   After:  {after_m['search_success_rate']:.1%} (+{sr_m_a:.1f}%p)
   총 개선: +{sr_total:.1f}%p

3. 평균 응답 시간
   Before: {before_m['avg_response_time']:.2f}초
   Middle: {middle_m['avg_response_time']:.2f}초
   After:  {after_m['avg_response_time']:.2f}초

{'='*70}
개선 기여도 분석
{'='*70}

문서 최적화 (Before → Middle):
   • 키워드 정확도: +{kw_b_m:.1f}%p
   • 전체 기여율: {doc_contribution:.1f}%
   • 주요 변경: 동의어 30개 추가, 문서 구조 개선

코드 최적화 (Middle → After):
   • 키워드 정확도: +{kw_m_a:.1f}%p
   • 전체 기여율: {code_contribution:.1f}%
   • 주요 변경: chunk 2000으로 확대, 동의어 50+ 확장

{'='*70}
핵심 개선 사항
{'='*70}

Stage 1 (Before → Middle): 문서 최적화
   ✅ 동의어 사전 구축 (0 → 30개)
   ✅ MD 파일 구조 개선
   ✅ Chunk 크기 확대 (800 → 1200)

Stage 2 (Middle → After): 코드 최적화
   ✅ 동의어 대폭 확장 (30 → 50+개)
   ✅ Chunk 크기 최적화 (1200 → 2000)
   ✅ 프롬프트 상세화 (표 완전 포함 명시)
   ✅ 대한항공 문서 통합 (3개 → 1개)

{'='*70}
"""

    with open('evaluation_3stages_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print("📄 저장: evaluation_3stages_report.txt")


def main():
    """메인 실행"""
    print("="*70)
    print("📊 3단계 평가 결과 시각화")
    print("="*70)

    # 최근 결과 파일 찾기
    result_files = list(Path('.').glob('evaluation_3stages_*.json'))

    if not result_files:
        print("❌ 3단계 평가 결과 파일을 찾을 수 없습니다.")
        print("먼저 evaluate_3stages.py를 실행하세요.")
        return

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 결과 파일: {latest_file}")

    # 데이터 로드
    data = load_results(latest_file)
    before_m = data['before']['metrics']
    middle_m = data['middle']['metrics']
    after_m = data['after']['metrics']

    print("\n📊 그래프 생성 중...")

    # 그래프 생성
    plot_3stage_keyword_accuracy(before_m, middle_m, after_m)
    plot_3stage_search_success(before_m, middle_m, after_m)
    plot_improvement_breakdown(before_m, middle_m, after_m)
    plot_response_time_comparison(before_m, middle_m, after_m)

    # 텍스트 리포트
    print("\n📄 텍스트 리포트 생성 중...")
    generate_3stage_report(data)

    print("\n✅ 모든 시각화 완료!")
    print("\n생성된 파일:")
    print("   • 3stage_keyword_accuracy.png")
    print("   • 3stage_search_success.png")
    print("   • improvement_breakdown.png")
    print("   • 3stage_response_time.png")
    print("   • evaluation_3stages_report.txt")


if __name__ == "__main__":
    main()