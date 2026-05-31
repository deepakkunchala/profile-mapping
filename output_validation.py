"""
================================================================================
                     MATCHING ENGINE - HELPER UTILITIES
                    Testing, Validation, and Analysis Tools
================================================================================

This module provides utility functions for:
  - Testing the matching engine
  - Validating results
  - Analyzing match quality
  - Generating reports

Usage:
  python helper_utilities.py

Dependencies:
  pandas, numpy, openpyxl

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


# =============================================================================
#                          VALIDATION FUNCTIONS
# =============================================================================

def validate_output_structure(final_output_file: str = "final_output.xlsx") -> Dict[str, bool]:
    """
    Validate that output file has correct structure and columns.
    
    Args:
        final_output_file: Path to final_output.xlsx
    
    Returns:
        Dict with validation results
    """
    print("\n" + "="*80)
    print("VALIDATING OUTPUT STRUCTURE")
    print("="*80)
    
    try:
        df = pd.read_excel(final_output_file)
    except FileNotFoundError:
        print(f"✗ File not found: {final_output_file}")
        return {'file_exists': False}
    
    results = {'file_exists': True}
    
    required_columns = [
        'J_ID', 'Junior_Name', 'Junior_Email',
        'R1_Senior_ID', 'R1_Senior_Name', 'R1_Score',
        'R2_Senior_ID', 'R2_Senior_Name', 'R2_Score',
        'Manual_Review'
    ]
    
    for col in required_columns:
        if col in df.columns:
            results[f'has_{col}'] = True
        else:
            results[f'has_{col}'] = False
            print(f"  ✗ Missing column: {col}")
    
    # Check data consistency
    results['all_juniors_matched_r1'] = df['R1_Senior_ID'].notna().all()
    results['all_juniors_matched_r2'] = df['R2_Senior_ID'].notna().all()
    results['no_duplicate_juniors'] = len(df) == len(df['J_ID'].unique())
    
    if not results['all_juniors_matched_r1']:
        print(f"  ✗ {df['R1_Senior_ID'].isna().sum()} juniors not matched in R1")
    
    if not results['all_juniors_matched_r2']:
        print(f"  ✗ {df['R2_Senior_ID'].isna().sum()} juniors not matched in R2")
    
    if not results['no_duplicate_juniors']:
        print(f"  ✗ Duplicate juniors found!")
    
    if all(results.values()):
        print("✓ Output structure is valid!")
    
    return results


def validate_senior_loads(final_output_file: str = "final_output.xlsx", 
                         max_cap: int = 3) -> pd.DataFrame:
    """
    Validate senior load distribution.
    
    Args:
        final_output_file: Path to final_output.xlsx
        max_cap: Maximum allowed load per senior
    
    Returns:
        DataFrame with load analysis
    """
    print("\n" + "="*80)
    print("VALIDATING SENIOR LOAD DISTRIBUTION")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    # Count loads
    r1_loads = df['R1_Senior_ID'].value_counts()
    r2_loads = df['R2_Senior_ID'].value_counts()
    
    # Combine
    all_seniors = set(r1_loads.index).union(set(r2_loads.index))
    
    load_data = []
    for senior_id in sorted(all_seniors):
        r1_count = r1_loads.get(senior_id, 0)
        r2_count = r2_loads.get(senior_id, 0)
        total_load = r1_count + r2_count
        
        # Check overload per-round independently
        r1_status = "OVERLOAD" if r1_count > max_cap else "NORMAL"
        r2_status = "OVERLOAD" if r2_count > max_cap else "NORMAL"
        # Overall status is OVERLOAD if either round is overloaded
        status = "OVERLOAD" if (r1_status == "OVERLOAD" or r2_status == "OVERLOAD") else "NORMAL"
        
        load_data.append({
            'Senior_ID': senior_id,
            'R1_Count': r1_count,
            'R2_Count': r2_count,
            'Total_Load': total_load,
            'Max_Cap': max_cap,
            'R1_Status': r1_status,
            'R2_Status': r2_status,
            'Status': status
        })
    
    load_df = pd.DataFrame(load_data).sort_values('Total_Load', ascending=False)
    
    print(f"\nTotal seniors: {len(load_df)}")
    print(f"Normal loads: {len(load_df[load_df['Status'] == 'NORMAL'])}")
    
    overloaded_seniors = load_df[load_df['Status'] == 'OVERLOAD']
    print(f"Overloaded (per-round): {len(overloaded_seniors)}")
    if len(overloaded_seniors) > 0:
        print(f"  Overloaded Senior IDs: {', '.join(overloaded_seniors['Senior_ID'].astype(str).tolist())}")
        for _, row in overloaded_seniors.iterrows():
            r1_flag = "(OVERLOADED)" if row['R1_Status'] == "OVERLOAD" else ""
            r2_flag = "(OVERLOADED)" if row['R2_Status'] == "OVERLOAD" else ""
            print(f"    {row['Senior_ID']}: R1={row['R1_Count']}{r1_flag}, R2={row['R2_Count']}{r2_flag}, Total={row['Total_Load']}")
    
    print(f"\nAverage load: {load_df['Total_Load'].mean():.2f}")
    print(f"Max load: {load_df['Total_Load'].max()}")
    print(f"Min load: {load_df['Total_Load'].min()}")
    print(f"Std dev: {load_df['Total_Load'].std():.2f}")
    
    # Per-round analysis
    r1_overloaded = len(load_df[load_df['R1_Status'] == 'OVERLOAD'])
    r2_overloaded = len(load_df[load_df['R2_Status'] == 'OVERLOAD'])
    print(f"\nPer-Round Overload:")
    print(f"  R1 Overloaded: {r1_overloaded}")
    print(f"  R2 Overloaded: {r2_overloaded}")
    
    # Fairness metric
    if len(load_df) > 1:
        fairness_ratio = load_df['Total_Load'].std() / load_df['Total_Load'].mean()
        print(f"\nFairness ratio (std/mean): {fairness_ratio:.3f}")
        if fairness_ratio < 0.3:
            print("  ✓ EXCELLENT fairness (very balanced)")
        elif fairness_ratio < 0.5:
            print("  ✓ GOOD fairness")
        elif fairness_ratio < 0.7:
            print("  ⚠ MODERATE fairness (some imbalance)")
        else:
            print("  ✗ POOR fairness (significant imbalance)")
    
    return load_df


def analyze_confidence_distribution(final_output_file: str = "final_output.xlsx"):
    """
    Analyze confidence level distribution.
    
    Args:
        final_output_file: Path to final_output.xlsx
    """
    print("\n" + "="*80)
    print("ANALYZING CONFIDENCE DISTRIBUTION")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    print("\nROUND 1 Confidence:")
    r1_conf = df['R1_Confidence'].value_counts()
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        count = r1_conf.get(level, 0)
        pct = (count / len(df)) * 100
        print(f"  {level:6} : {count:4} ({pct:5.1f}%)")
    
    print("\nROUND 2 Confidence:")
    r2_conf = df['R2_Confidence'].value_counts()
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        count = r2_conf.get(level, 0)
        pct = (count / len(df)) * 100
        print(f"  {level:6} : {count:4} ({pct:5.1f}%)")
    
    # Check for HIGH confidence in both
    both_high = len(df[(df['R1_Confidence'] == 'HIGH') & (df['R2_Confidence'] == 'HIGH')])
    pct = (both_high / len(df)) * 100
    print(f"\nBOTH rounds HIGH confidence: {both_high} ({pct:.1f}%)")
    
    # Check for LOW confidence in either
    either_low = len(df[(df['R1_Confidence'] == 'LOW') | (df['R2_Confidence'] == 'LOW')])
    pct = (either_low / len(df)) * 100
    print(f"EITHER round LOW confidence: {either_low} ({pct:.1f}%)")


def analyze_manual_review_bucket(final_output_file: str = "final_output.xlsx"):
    """
    Analyze and categorize manual review cases.
    
    Args:
        final_output_file: Path to final_output.xlsx
    """
    print("\n" + "="*80)
    print("ANALYZING MANUAL REVIEW BUCKET")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    manual_review = df[df['Manual_Review'] == True]
    
    print(f"\nTotal for manual review: {len(manual_review)} ({len(manual_review)/len(df)*100:.1f}%)")
    
    if len(manual_review) == 0:
        print("✓ No matches require manual review!")
        return
    
    # Analyze reasons
    reason_counts = {}
    for reasons_str in manual_review['Review_Reasons']:
        if pd.isna(reasons_str):
            continue
        reasons = [r.strip() for r in str(reasons_str).split(';')]
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print("\nReview Reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(manual_review)) * 100
        print(f"  {reason:30} : {count:4} ({pct:5.1f}%)")


def check_same_senior_both_rounds(final_output_file: str = "final_output.xlsx"):
    """
    Check how many juniors have same senior in both rounds.
    
    Args:
        final_output_file: Path to final_output.xlsx
    """
    print("\n" + "="*80)
    print("CHECKING SAME SENIOR BOTH ROUNDS")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    same_senior = df[df['Same_Senior_R1_R2'] == True]
    
    print(f"\nJuniors with same senior both rounds: {len(same_senior)} ({len(same_senior)/len(df)*100:.1f}%)")
    
    if len(same_senior) == 0:
        print("✓ All juniors have different seniors (ideal!)")
    elif len(same_senior) <= len(df) * 0.05:
        print("✓ Acceptable (<5%)")
    else:
        print("⚠ High reuse rate (>5%), consider reviewing segment constraints")


# =============================================================================
#                          REPORT GENERATION
# =============================================================================

def generate_summary_report(
    final_output_file: str = "final_output.xlsx",
    output_file: str = "matching_summary_report.txt"
):
    """
    Generate comprehensive summary report.
    
    Args:
        final_output_file: Path to final_output.xlsx
        output_file: Path for output report
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MENTORSHIP MATCHING ENGINE - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Basic stats
        f.write("BASIC STATISTICS\n")
        f.write("─" * 80 + "\n")
        f.write(f"Total Juniors Matched: {len(df)}\n")
        f.write(f"Total Unique Seniors: {df['R1_Senior_ID'].nunique() + df['R2_Senior_ID'].nunique()}\n")
        f.write(f"Average R1 Score: {df['R1_Score'].mean():.2f} / 100\n")
        f.write(f"Average R2 Score: {df['R2_Score'].mean():.2f} / 80\n\n")
        
        # Confidence distribution
        f.write("CONFIDENCE DISTRIBUTION\n")
        f.write("─" * 80 + "\n")
        r1_conf = df['R1_Confidence'].value_counts()
        r2_conf = df['R2_Confidence'].value_counts()
        
        f.write("Round 1:\n")
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = r1_conf.get(level, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            f.write(f"  {level:8} : {count:4} ({pct:5.1f}%)\n")
        
        f.write("\nRound 2:\n")
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = r2_conf.get(level, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            f.write(f"  {level:8} : {count:4} ({pct:5.1f}%)\n")
        
        # Load distribution
        f.write("\n" + "="*80 + "\n")
        f.write("SENIOR LOAD DISTRIBUTION\n")
        f.write("─" * 80 + "\n")
        
        r1_loads = df['R1_Senior_ID'].value_counts()
        r2_loads = df['R2_Senior_ID'].value_counts()
        all_seniors = set(r1_loads.index).union(set(r2_loads.index))
        
        loads = [r1_loads.get(s, 0) + r2_loads.get(s, 0) for s in all_seniors]
        
        f.write(f"Total Seniors: {len(all_seniors)}\n")
        f.write(f"Average Load: {np.mean(loads):.2f}\n")
        f.write(f"Min Load: {min(loads)}\n")
        f.write(f"Max Load: {max(loads)}\n")
        f.write(f"Std Dev: {np.std(loads):.2f}\n")
        
        # Per-round overload check
        r1_overloaded = sum(1 for s in all_seniors if r1_loads.get(s, 0) > 3)
        r2_overloaded = sum(1 for s in all_seniors if r2_loads.get(s, 0) > 3)
        f.write(f"\nPer-Round Overload (>3):\n")
        f.write(f"  R1 Overloaded: {r1_overloaded} / {len(all_seniors)} ({r1_overloaded/len(all_seniors)*100:.1f}%)\n")
        f.write(f"  R2 Overloaded: {r2_overloaded} / {len(all_seniors)} ({r2_overloaded/len(all_seniors)*100:.1f}%)\n")
        
        # Manual review
        f.write("\n" + "="*80 + "\n")
        f.write("MANUAL REVIEW BUCKET\n")
        f.write("─" * 80 + "\n")
        
        manual = df[df['Manual_Review'] == True]
        f.write(f"Total for Review: {len(manual)} ({len(manual)/len(df)*100:.1f}%)\n\n")
        
        reason_counts = {}
        for reasons_str in manual['Review_Reasons']:
            if pd.isna(reasons_str):
                continue
            reasons = [r.strip() for r in str(reasons_str).split(';')]
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(manual)) * 100 if len(manual) > 0 else 0
            f.write(f"  {reason:30} : {count:4} ({pct:5.1f}%)\n")
        
        # Same senior both rounds
        f.write("\n" + "="*80 + "\n")
        f.write("SAME SENIOR BOTH ROUNDS\n")
        f.write("─" * 80 + "\n")
        
        same = len(df[df['Same_Senior_R1_R2'] == True])
        f.write(f"Total: {same} ({same/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Report generated successfully!\n")
        f.write(f"Saved to: {output_file}\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved to: {output_file}")


def export_for_stakeholders(
    final_output_file: str = "final_output.xlsx",
    output_file: str = "stakeholder_export.xlsx"
):
    """
    Export results in stakeholder-friendly format.
    
    Args:
        final_output_file: Path to final_output.xlsx
        output_file: Path for stakeholder export
    """
    print("\n" + "="*80)
    print("GENERATING STAKEHOLDER EXPORT")
    print("="*80)
    
    df = pd.read_excel(final_output_file)
    
    # For juniors: their matches
    juniors_view = df[[
        'J_ID', 'Junior_Name',
        'R1_Senior_Name',
        'R2_Senior_Name'
    ]].copy()
    juniors_view.columns = [
        'Junior_ID', 'Junior_Name',
        'Mentor_1_Name',
        'Mentor_2_Name'
    ]
    
    # For seniors: their matches
    r1_dict = df[['R1_Senior_ID', 'Junior_Name']].copy()
    r1_dict.columns = ['Senior_ID', 'Junior_Name']
    r1_dict['Round'] = 'R1'
    
    r2_dict = df[['R2_Senior_ID', 'Junior_Name']].copy()
    r2_dict.columns = ['Senior_ID', 'Junior_Name']
    r2_dict['Round'] = 'R2'
    
    seniors_view = pd.concat([r1_dict, r2_dict], ignore_index=True)
    seniors_view = seniors_view.sort_values('Senior_ID')
    
    # Save with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        juniors_view.to_excel(writer, sheet_name='For_Juniors', index=False)
        seniors_view.to_excel(writer, sheet_name='For_Seniors', index=False)
        df.to_excel(writer, sheet_name='Full_Details', index=False)
    
    print(f"✓ Stakeholder export saved to: {output_file}")


# =============================================================================
#                          QUICK VALIDATION SUITE
# =============================================================================

def run_quick_validation():
    """
    Run all validation checks quickly.
    """
    print("\n")
    print("█" * 80)
    print("█" + "  QUICK VALIDATION SUITE".center(78) + "█")
    print("█" * 80)
    
    # Check if files exist
    if not Path("final_output.xlsx").exists():
        print("\n✗ final_output.xlsx not found!")
        print("  Run matching_engine.py first")
        return
    
    # Run validations
    validate_output_structure()
    analyze_confidence_distribution()
    validate_senior_loads()
    check_same_senior_both_rounds()
    analyze_manual_review_bucket()
    
    # Generate reports
    generate_summary_report()
    export_for_stakeholders()
    
    print("\n" + "█" * 80)
    print("█" + "  ✓ VALIDATION COMPLETE".center(78) + "█")
    print("█" * 80)
    print("\nGenerated files:")
    print("  1. matching_summary_report.txt")
    print("  2. stakeholder_export.xlsx")
    print("\n")


if __name__ == "__main__":
    run_quick_validation()
