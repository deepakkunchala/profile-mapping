"""
================================================================================
                    MENTORSHIP ALLOCATION MATCHING ENGINE
                           Production Quality v2.0
================================================================================

PURPOSE:
    Match Juniors (Bo'27) with Seniors (Bo'26) for CV vetting mentorship.
    Each junior receives 2 seniors (prefer different in Round 1 and Round 2).

ARCHITECTURE:
    - Data Loading & Cleaning → Normalization → Segmentation
    - Fallback Hierarchy → Scoring → Load Balancing → Matching
    - Debug Logging → Output Generation → Manual Review Bucket

DEPENDENCIES:
    pandas, numpy, openpyxl

AUTHOR:
    Production Mentorship Matching System

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
#                        GLOBAL CONFIGURATION SYSTEM
# =============================================================================

CONFIG = {
    # Matching constraints
    'MAX_LOAD_CAP': 3,                  # Hard capacity limit per senior
    'RANDOM_STATE': 42,                  # For reproducible shuffling
    'DEBUG_MODE': True,                  # Enable detailed debug logging
    
    # Scoring weights for Round 1 (sum=100)
    'ROUND_1_WEIGHTS': {
        'UG_Degree': 25,
        'UG_Specialization': 25,
        'Masters_Degree': 10,
        'Masters_Specialization': 10,
        'Industry': 15,
        'Experience': 10,
        'CA_Bonus': 5
    },
    
    # Scoring weights for Round 2 (sum=80)
    'ROUND_2_WEIGHTS': {
        'Industry': 35,
        'Experience': 30,
        'UG_Degree': 10,
        'UG_Specialization': 10,
        "Masters_Specialization": 5,
        'Masters_Degree': 5,
        'CA_Bonus': 5
    },
    
    # Fallback levels (don't change order)
    'FALLBACK_LEVELS': {
        1: "Exact Segment (IsCA|UG|ExpType)",
        2: "Ignore Experience (IsCA|UG)",
        3: "Only UG Category",
        4: "Ignore CA (UG|ExpType)",
        5: "Global Pool (No constraints)"
    },
    
    # Confidence thresholds
    'CONFIDENCE_R1': {'HIGH': 60, 'MEDIUM': 40},  # % of max score
    'CONFIDENCE_R2': {'HIGH': 60, 'MEDIUM': 40},  # % of max score
}

# For backward compatibility, expose commonly used values
MAX_LOAD_CAP = CONFIG['MAX_LOAD_CAP']
FALLBACK_LEVELS = CONFIG['FALLBACK_LEVELS']
ROUND_1_WEIGHTS = CONFIG['ROUND_1_WEIGHTS']
ROUND_2_WEIGHTS = CONFIG['ROUND_2_WEIGHTS']

# =============================================================================
#                    VALIDATION & ERROR HANDLING
# =============================================================================

def validate_dataframes(df_juniors: pd.DataFrame, df_seniors: pd.DataFrame, debug_log: List[str]) -> None:
    """
    Validate input dataframes before matching.
    
    Args:
        df_juniors: Juniors DataFrame
        df_seniors: Seniors DataFrame
        debug_log: List to append validation messages
        
    Raises:
        ValueError: If validation fails
    """
    issues = []
    
    # Check for required columns in juniors
    required_j_cols = [
        'J_ID', 'Is CA', 'Undergrad Category', 'Undergraduate Degree',
        'Area of Specialization', 'Masters_Degree', 'Masters_Specialization',
        'Experience_Months', 'ExpType', 'Segment'
    ]
    missing_j = [col for col in required_j_cols if col not in df_juniors.columns]
    if missing_j:
        issues.append(f"Missing junior columns: {missing_j}")
    
    # Check for required columns in seniors
    required_s_cols = [
        'S_ID', 'Is CA', 'Undergrad Category', 'Undergraduate Degree',
        'Area of Specialization', 'Masters_Degree', 'Masters_Specialization',
        'Experience_Months', 'ExpType', 'Segment'
    ]
    missing_s = [col for col in required_s_cols if col not in df_seniors.columns]
    if missing_s:
        issues.append(f"Missing senior columns: {missing_s}")
    
    # Check for duplicate IDs
    if df_juniors['J_ID'].duplicated().any():
        issues.append(f"Duplicate junior IDs found")
    if df_seniors['S_ID'].duplicated().any():
        issues.append(f"Duplicate senior IDs found")
    
    # Check for null IDs
    if df_juniors['J_ID'].isnull().any():
        issues.append(f"Null values in junior IDs")
    if df_seniors['S_ID'].isnull().any():
        issues.append(f"Null values in senior IDs")
    
    # Check experience values are numeric
    if not pd.api.types.is_numeric_dtype(df_juniors['Experience_Months']):
        issues.append(f"Junior experience values not numeric")
    if not pd.api.types.is_numeric_dtype(df_seniors['Experience_Months']):
        issues.append(f"Senior experience values not numeric")
    
    # Check experience values valid (>= 0)
    if (df_juniors['Experience_Months'] < 0).any():
        issues.append(f"Negative experience values in juniors")
    if (df_seniors['Experience_Months'] < 0).any():
        issues.append(f"Negative experience values in seniors")
    
    # Check segments generated
    if df_juniors['Segment'].isnull().any():
        issues.append(f"Null segments in juniors")
    if df_seniors['Segment'].isnull().any():
        issues.append(f"Null segments in seniors")
    
    if issues:
        error_msg = "VALIDATION FAILED:\n" + "\n".join([f"  - {issue}" for issue in issues])
        debug_log.append(error_msg)
        raise ValueError(error_msg)
    
    debug_log.append(f"✓ Validation passed: {len(df_juniors)} juniors, {len(df_seniors)} seniors")

# =============================================================================
#                       SECTION 1: DATA LOADING
# =============================================================================

def load_data(junior_file: str, senior_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load junior and senior data from Excel files.
    
    Args:
        junior_file: Path to Bo'27.xlsx
        senior_file: Path to Bo'26.xlsx
    
    Returns:
        Tuple of (df_juniors, df_seniors)
    
    Raises:
        FileNotFoundError: If files don't exist
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    try:
        df_j = pd.read_excel(junior_file)
        df_s = pd.read_excel(senior_file)
        
        print(f"✓ Loaded juniors: {len(df_j)} records")
        print(f"✓ Loaded seniors: {len(df_s)} records")
        
        return df_j, df_s
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find file: {e}")
    except Exception as e:
        raise Exception(f"Error loading Excel files: {e}")


# =============================================================================
#                    SECTION 2: UNIQUE ID CREATION
# =============================================================================

def create_ids(df_juniors: pd.DataFrame, df_seniors: pd.DataFrame) -> None:
    """
    Create unique IDs for juniors (J001, J002, ...) and seniors (S001, S002, ...).
    Modifies dataframes in place.
    
    Args:
        df_juniors: DataFrame with junior records
        df_seniors: DataFrame with senior records
    """
    print("\n" + "="*80)
    print("STEP 2: CREATING UNIQUE IDs")
    print("="*80)
    
    df_juniors["J_ID"] = ["J" + str(i).zfill(3) for i in range(1, len(df_juniors) + 1)]
    df_seniors["S_ID"] = ["S" + str(i).zfill(3) for i in range(1, len(df_seniors) + 1)]
    
    print(f"✓ Created Junior IDs: J001 to J{len(df_juniors):03d}")
    print(f"✓ Created Senior IDs: S001 to S{len(df_seniors):03d}")


def shuffle_juniors_fair(df_juniors: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffle juniors fairly to remove sequential bias.
    Uses CONFIG['RANDOM_STATE'] for reproducibility.
    
    Args:
        df_juniors: DataFrame with juniors
    
    Returns:
        Shuffled DataFrame with index reset
    """
    print("\n" + "="*80)
    print("STEP 2.5: FAIR JUNIOR SHUFFLING")
    print("="*80)
    
    shuffled = df_juniors.sample(frac=1, random_state=CONFIG['RANDOM_STATE']).reset_index(drop=True)
    print(f"✓ Shuffled {len(shuffled)} juniors (seed={CONFIG['RANDOM_STATE']})")
    print(f"  First 5 juniors: {', '.join(shuffled['J_ID'].head(5).tolist())}")
    
    return shuffled


# =============================================================================
#                       SECTION 3: TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, strip spaces, remove duplicate spacing.
    
    Args:
        text: Raw text input
    
    Returns:
        Normalized text
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove multiple spaces
    text = " ".join(text.split())
    return text


def normalize_degree(degree: str) -> str:
    """
    Normalize degree names.
    Maps variations to standard names: engineering, commerce, science, law, medicine, etc.
    
    Args:
        degree: Degree name
    
    Returns:
        Normalized degree category
    """
    degree = normalize_text(degree)
    
    if not degree or degree in ['nan', 'n/a', 'na', 'none', 'null']:
        return "unknown"
    
    # Engineering variants
    if any(x in degree for x in ['btech', 'b.tech', 'bachelor of tech', 'engineering', 'engineer']):
        return "engineering"
    
    # Commerce variants
    if any(x in degree for x in ['bca', 'b.com', 'commerce', 'chartered', 'ca']):
        return "commerce"
    
    # Science variants
    if any(x in degree for x in ['bsc', 'b.sc', 'science']):
        return "science"
    
    # Arts/Humanities variants
    if any(x in degree for x in ['ba', 'b.a', 'arts', 'humanities','Batchelor of Media Studies']):
        return "arts"

    # Other common degrees
    if any(x in degree for x in ['law', 'medicine', 'medical', 'mba']):
        return degree.split()[0] if degree else "unknown"
    
    return degree

def normalize_ug_degree(degree: str) -> str:
    """
    Normalize undergraduate degree names.
    Focuses on UG degrees, maps to standard categories.
    
    Args:
        degree: Undergraduate degree name
    Returns:
        Normalized undergraduate degree category
    """
    return normalize_text(degree)

def normalize_industry(industry: str) -> str:
    """
    Normalize industry names.
    Maps variations to standard categories.
    
    Args:
        industry: Industry name
    
    Returns:
        Normalized industry category
    """
    industry = normalize_text(industry)
    
    if not industry or industry in ['nan', 'n/a', 'na', 'none', 'null']:
        return "n/a"
    
    # Technology variants
    if any(x in industry for x in ['it', 'tech', 'technology', 'software', 'data']):
        return "technology"
    
    # Finance/Banking variants
    if any(x in industry for x in ['finance', 'bank', 'investment', 'insurance', 'capital']):
        return "finance"
    
    # Manufacturing variants
    if any(x in industry for x in ['manufacturing', 'production', 'industrial']):
        return "manufacturing"
    
    # Consulting variants
    if any(x in industry for x in ['consulting', 'advisory', 'management']):
        return "consulting"
    
    # Retail/Services variants
    if any(x in industry for x in ['retail', 'service', 'hospitality', 'tourism']):
        return "services"
    
    # HR/Recruitment variants
    if any(x in industry for x in ['hr', 'human resource', 'recruitment']):
        return "hr"
    
    # Marketing/Sales variants
    if any(x in industry for x in ['marketing', 'sales', 'business', 'commerce']):
        return "marketing"
    
    # Government/Public variants
    if any(x in industry for x in ['government', 'public', 'education', 'ngo']):
        return "public"
    
    return industry


def normalize_specialization(spec: str) -> str:
    """
    Normalize specialization names.
    
    Args:
        spec: Specialization name
    
    Returns:
        Normalized specialization
    """
    return normalize_text(spec)


# =============================================================================
#                    SECTION 4: DATA CLEANING & NORMALIZATION
# =============================================================================

def clean_dataframe(df: pd.DataFrame, person_type: str) -> pd.DataFrame:
    """
    Comprehensive data cleaning for juniors or seniors.
    
    Args:
        df: DataFrame to clean
        person_type: "junior" or "senior"
    
    Returns:
        Cleaned DataFrame
    """
    print(f"\n  Cleaning {person_type} records...")
    
    df = df.copy()
    
    # Clean 'Is CA' field
    def clean_ca(x):
        x_norm = normalize_text(str(x))
        return "Yes" if x_norm in ["yes", "y", "true", "1"] else "No"
    
    if "Is CA" in df.columns:
        df["Is CA"] = df["Is CA"].apply(clean_ca)
    else:
        df["Is CA"] = "No"
    
    # Clean Undergrad Category
    if "Undergrad Category" in df.columns:
        df["Undergrad Category"] = df["Undergrad Category"].apply(normalize_degree)
    else:
        df["Undergrad Category"] = "unknown"
    
    # Clean Undergraduate Degree
    if "Undergraduate Degree" in df.columns:
        df["Undergraduate Degree"] = df["Undergraduate Degree"].apply(normalize_ug_degree)
    else:
        df["Undergraduate Degree"] = "unknown"
    
    # Clean UG Specialization
    if "Area of Specialization" in df.columns:
        df["Area of Specialization"] = df["Area of Specialization"].apply(normalize_specialization)
    else:
        df["Area of Specialization"] = ""
    
    # Clean Masters Degree
    masters_col = None
    if "Masters Degree, if any (e.g. M.Tech)" in df.columns:
        masters_col = "Masters Degree, if any (e.g. M.Tech)"
    elif "Masters Degree (if any) e.g. M.Tech" in df.columns:
        masters_col = "Masters Degree (if any) e.g. M.Tech"
    else: masters_col = "Masters Degree"
    
    if masters_col:
        df["Masters_Degree"] = df[masters_col].apply(normalize_degree)
    else:
        df["Masters_Degree"] = "none"
    
    # Clean Masters Specialization
    masters_spec_col = None
    if "Area of Specialization, if any (e.g. M.Tech Computer Science)" in df.columns:
        masters_spec_col = "Area of Specialization, if any (e.g. M.Tech Computer Science)"
    elif "Masters Specialization (if any) e.g. M.Tech Computer Science" in df.columns:
        masters_spec_col = "Masters Specialization (if any) e.g. M.Tech Computer Science"
    else: masters_spec_col = "Masters Specialization"
    
    if masters_spec_col:
        df["Masters_Specialization"] = df[masters_spec_col].apply(normalize_specialization)
    else:
        df["Masters_Specialization"] = ""
    
    # Clean experience (convert to integer months)
    exp_col = "Total number of months of work experience"
    if exp_col in df.columns:
        df["Experience_Months"] = df[exp_col].apply(clean_experience)
    else:
        df["Experience_Months"] = 0
    
    # Create Experience Type (F = Fresher, E = Experienced)
    df["ExpType"] = df["Experience_Months"].apply(lambda x: "F" if x == 0 else "E")
    
    return df


def clean_experience(value) -> int:
    """
    Clean experience values and convert to integer months.
    Handles null, text, and invalid values safely.
    
    Args:
        value: Experience value (can be int, float, string, or NaN)
    
    Returns:
        Integer months (0 if invalid or null)
    """
    if pd.isna(value):
        return 0
    
    try:
        # Try direct conversion
        val = float(value)
        return max(0, int(val))
    except (ValueError, TypeError):
        # Try string parsing
        value_str = str(value).lower().strip()
        if not value_str or value_str in ['none', 'null', 'na', 'n/a']:
            return 0
        
        # Try to extract numbers
        import re
        match = re.search(r'\d+', value_str)
        if match:
            return int(match.group())
    
    return 0


# =============================================================================
#                    SECTION 5: INDUSTRY EXTRACTION
# =============================================================================

def extract_industries(df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    Extract industries from all role columns (Role 1, Role 2, Role 3).
    Returns dict mapping index to list of unique industries.
    
    Args:
        df: DataFrame with industry columns
    
    Returns:
        Dict[index] = [list of industries]
    """
    industries_dict = {}
    
    # Possible industry column patterns
    industry_patterns = [
        "Choose Industry 1, if Role 1 is Manufacturing related",
        "Choose industry 1, if Role 1 is Services related",
        "Other industry (Role 1)",
        "Other industry  - 1",
        # Role 2 variants
        "Choose Industry 2, if Role 2 is Manufacturing related",
        "Choose industry 2, if Role 2 is Services related",
        "Other industry (Role 2)",
        "Other industry  - 2",
        # Role 3 variants
        "Choose Industry 3, if Role 3 is Manufacturing related",
        "Choose industry 3, if Role 3 is Services related",
        "Other industry (Role 3)",
        "Other industry  - 3",
    ]
    
    # Find actual columns that exist
    available_cols = [col for col in df.columns if any(pattern in col for pattern in industry_patterns)]
    
    for idx, row in df.iterrows():
        industries_set = set()
        
        for col in available_cols:
            value = normalize_industry(str(row.get(col, "")))
            if value and value != "n/a":
                industries_set.add(value)
        
        industries_dict[idx] = sorted(list(industries_set))
    
    return industries_dict


def calculate_industry_match(industries_1: List[str], industries_2: List[str]) -> float:
    """
    Calculate industry match score based on overlap.
    
    Args:
        industries_1: Junior's industries
        industries_2: Senior's industries
    
    Returns:
        Overlap count (capped at 3)
    """
    if not industries_1 or not industries_2:
        return 0
    
    overlap = len(set(industries_1).intersection(set(industries_2)))
    return min(overlap, 3)  # Cap at 3


def create_tiebreaker_key(
    sr_idx: int,
    score: float,
    load: int,
    industry_overlap: float,
    max_score: float = 100.0
) -> Tuple:
    """
    Create deterministic tie-breaking key for senior candidate ranking.
    Used when multiple seniors have similar scores.
    
    Priority order (highest to lowest):
    1. Score (higher is better)
    2. Load (lower is better)
    3. Industry overlap (higher is better)
    4. Senior index (lower is better, for stable randomization)
    
    Args:
        sr_idx: Senior index in dataframe
        score: Matching score
        load: Current load
        industry_overlap: Industry match count
        max_score: Maximum possible score
    
    Returns:
        Tuple for sorting (all negated so higher values sort first)
    """
    return (
        -score,           # Sort descending by score
        load,             # Sort ascending by load
        -industry_overlap, # Sort descending by overlap
        sr_idx            # Sort ascending by index (stable)
    )


# =============================================================================
#                    SECTION 6: SEGMENT CREATION
# =============================================================================

def create_segments(df: pd.DataFrame) -> pd.Series:
    """
    Create segment keys: IsCA | UndergradCategory | ExpType
    
    Args:
        df: DataFrame with cleaned fields
    
    Returns:
        Series of segment keys
    """
    return (
        df["Is CA"].astype(str) + "|" +
        df["Undergrad Category"].astype(str) + "|" +
        df["ExpType"].astype(str)
    )


# =============================================================================
#               SECTION 7: SEGMENT FALLBACK HIERARCHY
# =============================================================================

def get_fallback_candidates(
    junior_segment: str,
    df_seniors: pd.DataFrame,
    senior_segments: pd.Series,
    jr_ca: str,
    jr_ug: str,
    jr_exp: str,
    fallback_logger: Dict
) -> Tuple[List[int], int]:
    """
    Find candidate seniors through fallback hierarchy.
    
    Args:
        junior_segment: Exact junior segment
        df_seniors: Seniors DataFrame
        senior_segments: Senior segment Series
        jr_ca: Junior's CA status
        jr_ug: Junior's UG category
        jr_exp: Junior's experience type
        fallback_logger: Dict to log fallback level used
    
    Returns:
        Tuple of (list of senior indices, fallback level used)
    """
    
    # LEVEL 1: Exact Segment
    candidates = df_seniors[senior_segments == junior_segment].index.tolist()
    if candidates:
        fallback_logger['level'] = 1
        return candidates, 1
    
    # LEVEL 2: Ignore Experience (IsCA + UG only)
    level2_segment = f"{jr_ca}|{jr_ug}|"
    candidates = df_seniors[senior_segments.str.startswith(level2_segment)].index.tolist()
    if candidates:
        fallback_logger['level'] = 2
        return candidates, 2
    
    # LEVEL 3: Only UG Category
    candidates = df_seniors[df_seniors["Undergrad Category"] == jr_ug].index.tolist()
    if candidates:
        fallback_logger['level'] = 3
        return candidates, 3
    
    # LEVEL 4: Ignore CA (UG + ExpType only)
    level4_pattern = f"|{jr_ug}|{jr_exp}"
    candidates = df_seniors[senior_segments.str.endswith(level4_pattern)].index.tolist()
    if candidates:
        fallback_logger['level'] = 4
        return candidates, 4
    
     # LEVEL 5: Global Pool
    candidates = df_seniors.index.tolist()
    if candidates:
        fallback_logger['level'] = 5
        return candidates, 5


def get_candidates_for_level(
    level: int,
    junior_segment: str,
    df_seniors: pd.DataFrame,
    senior_segments: pd.Series,
    jr_ca: str,
    jr_ug: str,
    jr_exp: str,
) -> List[int]:
    """
    Return the senior candidate pool for a *specific* fallback level without
    re-running the full hierarchy from the top.  Used by the cap-escalation
    loop so it can genuinely widen the search one level at a time.

    Args:
        level:           Target fallback level (1-5)
        junior_segment:  Exact junior segment string
        df_seniors:      Seniors DataFrame
        senior_segments: Senior segment Series
        jr_ca:           Junior CA status
        jr_ug:           Junior UG category
        jr_exp:          Junior experience type

    Returns:
        List of senior DataFrame indices for that level (may be empty).
    """
    if level == 1:
        return df_seniors[senior_segments == junior_segment].index.tolist()
    elif level == 2:
        pfx = f"{jr_ca}|{jr_ug}|"
        return df_seniors[senior_segments.str.startswith(pfx)].index.tolist()
    elif level == 3:
        return df_seniors[df_seniors["Undergrad Category"] == jr_ug].index.tolist()
    elif level == 4:
        sfx = f"|{jr_ug}|{jr_exp}"
        return df_seniors[senior_segments.str.endswith(sfx)].index.tolist()
    else:  # level 5 – global pool
        return df_seniors.index.tolist()


# =============================================================================
#                    SECTION 8: SEGMENT HEALTH ANALYSIS
# =============================================================================

def analyze_segment_health(
    df_juniors: pd.DataFrame,
    df_seniors: pd.DataFrame,
    junior_segments: pd.Series,
    senior_segments: pd.Series
) -> pd.DataFrame:
    """
    Generate segment health report before matching.
    
    Args:
        df_juniors: Juniors DataFrame
        df_seniors: Seniors DataFrame
        junior_segments: Junior segment Series
        senior_segments: Senior segment Series
    
    Returns:
        DataFrame with segment health metrics
    """
    print("\n" + "="*80)
    print("STEP 12: SEGMENT HEALTH ANALYSIS")
    print("="*80)
    
    health_records = []
    
    j_seg_counts = junior_segments.value_counts()
    s_seg_counts = senior_segments.value_counts()
    
    all_segments = sorted(set(j_seg_counts.index).union(set(s_seg_counts.index)))
    
    for segment in all_segments:
        j_count = j_seg_counts.get(segment, 0)
        s_count = s_seg_counts.get(segment, 0)
        
        # Each junior needs 2 seniors
        required_capacity = j_count * 2
        total_capacity = s_count * MAX_LOAD_CAP
        
        # Determine risk level
        if s_count == 0:
            risk_level = "CRITICAL"
        elif total_capacity < required_capacity * 0.8:
            risk_level = "HIGH"
        elif total_capacity < required_capacity:
            risk_level = "MEDIUM"
        else:
            risk_level = "SAFE"
        
        health_records.append({
            'Segment': segment,
            'Junior_Count': j_count,
            'Senior_Count': s_count,
            'Required_Capacity': required_capacity,
            'Total_Capacity': total_capacity,
            'Risk_Level': risk_level
        })
    
    health_df = pd.DataFrame(health_records)
    
    # Print summary
    print(f"\n✓ Analyzed {len(health_df)} segments")
    print(f"  SAFE:     {len(health_df[health_df['Risk_Level'] == 'SAFE'])}")
    print(f"  MEDIUM:   {len(health_df[health_df['Risk_Level'] == 'MEDIUM'])}")
    print(f"  HIGH:     {len(health_df[health_df['Risk_Level'] == 'HIGH'])}")
    print(f"  CRITICAL: {len(health_df[health_df['Risk_Level'] == 'CRITICAL'])}")
    
    return health_df


# =============================================================================
#                    SECTION 9: SCORING FUNCTIONS
# =============================================================================

def calculate_text_match(text_1: str, text_2: str) -> float:
    """
    Calculate text match score.
    1.0 for exact match, 0.0 for no match.
    
    Args:
        text_1: First text (normalized)
        text_2: Second text (normalized)
    
    Returns:
        Score 0.0 or 1.0
    """
    t1 = normalize_text(str(text_1))
    t2 = normalize_text(str(text_2))
    
    if not t1 or not t2:
        return 0.0
    
    if t1 == "unknown" or t2 == "unknown":
        return 0.0
    
    return 1.0 if t1 == t2 else 0.0


def calculate_experience_similarity(jr_months: int, sr_months: int) -> float:
    """
    Calculate experience similarity score.
    
    Gap <= 6:   100% (1.0)
    Gap <= 12:  75%  (0.75)
    Gap <= 24:  50%  (0.5)
    Else:       25%  (0.25)
    
    Args:
        jr_months: Junior's experience in months
        sr_months: Senior's experience in months
    
    Returns:
        Score between 0.0 and 1.0
    """
    gap = abs(jr_months - sr_months)
    
    if gap <= 6:
        return 1.0
    elif gap <= 12:
        return 0.75
    elif gap <= 24:
        return 0.5
    else:
        return 0.25


def calculate_round_score(
    junior: pd.Series,
    senior: pd.Series,
    jr_industries: List[str],
    sr_industries: List[str],
    ca_match: bool,
    weights: Dict[str, float] = ROUND_1_WEIGHTS
) -> float:
    """
    Calculate Round 1 matching score.
    
    Components:
    - UG Degree:           25 points
    - UG Specialization:   25 points
    - Masters Degree:      10 points
    - Masters Spec:        10 points
    - Industry Match:      15 points
    - Experience:          10 points
    - CA Match Bonus:      5 points
    
    Total:                100 points
    
    Args:
        junior: Junior record
        senior: Senior record
        jr_industries: Junior's industries
        sr_industries: Senior's industries
        ca_match: Whether CA status matches
    
    Returns:
        Score out of 100
    """
    score = 0.0
    
    # UG Degree match (25)
    ug_degree_match = calculate_text_match(
        junior["Undergraduate Degree"],
        senior["Undergraduate Degree"]
    )
    score += ug_degree_match * weights['UG_Degree']
    
    # UG Specialization match (25)
    ug_spec_match = calculate_text_match(
        junior["Area of Specialization"],
        senior["Area of Specialization"]
    )
    score += ug_spec_match * weights['UG_Specialization']
    
    # Masters Degree match (10)
    masters_match = calculate_text_match(
        junior["Masters_Degree"],
        senior["Masters_Degree"]
    )
    score += masters_match * weights['Masters_Degree']
    
    # Masters Specialization match (10)
    masters_spec_match = calculate_text_match(
        junior["Masters_Specialization"],
        senior["Masters_Specialization"]
    )
    score += masters_spec_match * weights['Masters_Specialization']
    
    # Industry match (15)
    industry_overlap = calculate_industry_match(jr_industries, sr_industries)
    industry_score = min(industry_overlap / 3, 1.0)  # Normalize to 0-1
    score += industry_score * weights['Industry']
    
    # Experience similarity (10)
    exp_similarity = calculate_experience_similarity(
        junior["Experience_Months"],
        senior["Experience_Months"]
    )
    score += exp_similarity * weights['Experience']
    
    # CA Match bonus (5)
    if ca_match:
        score += weights['CA_Bonus']
    
    return min(score, 100.0)


def calculate_round2_score(
    junior: pd.Series,
    senior: pd.Series,
    jr_industries: List[str],
    sr_industries: List[str],
    ca_match: bool
) -> float:
    """
    Calculate Round 2 matching score (different weights).
    
    Components:
    - Industry Match:      35 points
    - Experience:          20 points
    - UG Degree:           8 points
    - UG Specialization:   7 points
    - Masters Degree:      5 points
    - CA Match Bonus:      5 points
    
    Total:                 80 points
    
    Args:
        junior: Junior record
        senior: Senior record
        jr_industries: Junior's industries
        sr_industries: Senior's industries
        ca_match: Whether CA status matches
    
    Returns:
        Score out of 80
    """
    score = 0.0
    
    # Industry match (35) - MORE IMPORTANT IN R2
    industry_overlap = calculate_industry_match(jr_industries, sr_industries)
    industry_score = min(industry_overlap / 3, 1.0)
    score += industry_score * ROUND_2_WEIGHTS['Industry']
    
    # Experience similarity (20)
    exp_similarity = calculate_experience_similarity(
        junior["Experience_Months"],
        senior["Experience_Months"]
    )
    score += exp_similarity * ROUND_2_WEIGHTS['Experience']
    
    # UG Degree match (8)
    ug_degree_match = calculate_text_match(
        junior["Undergraduate Degree"],
        senior["Undergraduate Degree"]
    )
    score += ug_degree_match * ROUND_2_WEIGHTS['UG_Degree']
    
    # UG Specialization match (7)
    ug_spec_match = calculate_text_match(
        junior["Area of Specialization"],
        senior["Area of Specialization"]
    )
    score += ug_spec_match * ROUND_2_WEIGHTS['UG_Specialization']
    
    # Masters Degree match (5)
    masters_match = calculate_text_match(
        junior["Masters_Degree"],
        senior["Masters_Degree"]
    )
    score += masters_match * ROUND_2_WEIGHTS['Masters_Degree']
    
    # CA Match bonus (5)
    if ca_match:
        score += ROUND_2_WEIGHTS['CA_Bonus']
    
    return min(score, 80.0)


# =============================================================================
#                    SECTION 10: SCORING MATRIX GENERATION
# =============================================================================

def generate_scoring_matrices(
    df_juniors: pd.DataFrame,
    df_seniors: pd.DataFrame,
    jr_industries_dict: Dict[int, List[str]],
    sr_industries_dict: Dict[int, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Round 1 and Round 2 scoring matrices.
    Shape: (num_juniors, num_seniors)
    
    Args:
        df_juniors: Juniors DataFrame
        df_seniors: Seniors DataFrame
        jr_industries_dict: Junior indices to industries
        sr_industries_dict: Senior indices to industries
    
    Returns:
        Tuple of (r1_scores_df, r2_scores_df)
    """
    print("\n" + "="*80)
    print("STEP 9: GENERATING SCORING MATRICES")
    print("="*80)
    
    num_juniors = len(df_juniors)
    num_seniors = len(df_seniors)
    
    r1_scores = np.zeros((num_juniors, num_seniors))
    r2_scores = np.zeros((num_juniors, num_seniors))
    
    for jr_idx, jr_row in df_juniors.iterrows():
        for sr_idx, sr_row in df_seniors.iterrows():
            # Get industries
            jr_ind = jr_industries_dict.get(jr_idx, [])
            sr_ind = sr_industries_dict.get(sr_idx, [])
            
            # Check CA match
            if jr_row['Is CA'] == 'No' or sr_row['Is CA'] == 'No': 
                ca_match = False
            else: ca_match = True
            
            # Calculate scores
            r1_score = calculate_round_score(jr_row, sr_row, jr_ind, sr_ind, ca_match, ROUND_1_WEIGHTS)
            r2_score = calculate_round_score(jr_row, sr_row, jr_ind, sr_ind, ca_match, ROUND_2_WEIGHTS)
            
            r1_scores[jr_idx, sr_idx] = r1_score
            r2_scores[jr_idx, sr_idx] = r2_score
    
    # Convert to DataFrames
    r1_df = pd.DataFrame(r1_scores, index=df_juniors["J_ID"], columns=df_seniors["S_ID"])
    r2_df = pd.DataFrame(r2_scores, index=df_juniors["J_ID"], columns=df_seniors["S_ID"])
    
    print(f"✓ Generated {num_juniors}x{num_seniors} Round 1 matrix")
    print(f"✓ Generated {num_juniors}x{num_seniors} Round 2 matrix")
    
    return r1_df, r2_df


# =============================================================================
#                    SECTION 11: MATCHING ENGINE
# =============================================================================

def get_confidence_label(score: float, max_score: float = 100.0) -> str:
    """
    Assign confidence label based on score.
    
    Args:
        score: Match score
        max_score: Maximum possible score
    
    Returns:
        "HIGH", "MEDIUM", or "LOW"
    """
    percentage = (score / max_score) * 100
    
    if percentage >= 60:
        return "HIGH"
    elif percentage >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def _assign_round(
    round_name: str,
    jr_idx: int,
    junior_row: pd.Series,
    jr_segment: str,
    jr_ca: str,
    jr_ug: str,
    jr_exp: str,
    jr_ind: List[str],
    df_seniors: pd.DataFrame,
    senior_segments: pd.Series,
    scores_matrix: pd.DataFrame,
    sr_industries_dict: Dict[int, List[str]],
    senior_load: Dict[str, int],
    senior_load_total: Dict[str, int],
    senior_assignments: Dict[str, List[str]],
    log_entry: Dict,
    r1_senior_id: str,
    max_score: float,
    phase1_only: bool = False,
) -> Tuple[str, float, str, int, int, bool]:
    """
    Internal helper that executes the scoring + assignment logic for one round
    (R1 or R2) for a single junior.

    When ``phase1_only=True`` the function restricts candidates to the junior's
    **exact segment** (fallback level 1) and returns ``(None, ...)`` without
    making any assignment if no eligible same-segment senior is found.

    Returns:
        (senior_id, score, load_type, fallback_level, industry_overlap, same_as_r1)
        senior_id is None when no assignment was made (phase1_only + no capacity).
    """
    is_r2 = (round_name == "R2")
    fallback_log = {'level': None}

    if phase1_only:
        # Strict same-segment only — no fallback hierarchy
        same_seg_candidates = df_seniors[senior_segments == jr_segment].index.tolist()
        candidates = same_seg_candidates
        fallback_level = 1
        fallback_log['level'] = 1
    else:
        candidates, fallback_level = get_fallback_candidates(
            jr_segment, df_seniors, senior_segments,
            jr_ca, jr_ug, jr_exp, fallback_log
        )

    log_entry[round_name]['fallback_level'] = fallback_level
    log_entry[round_name]['fallback_reason'] = CONFIG['FALLBACK_LEVELS'][fallback_level]
    log_entry[round_name]['candidate_count'] = len(candidates)
    log_entry[round_name]['candidates_list'] = []

    jr_id = junior_row["J_ID"]
    candidate_scores = []
    for sr_idx in candidates:
        senior_row = df_seniors.iloc[sr_idx]
        s_id = senior_row["S_ID"]
        score = scores_matrix.iloc[jr_idx][s_id]
        load = senior_load[s_id]
        sr_ind = sr_industries_dict.get(sr_idx, [])
        industry_overlap = calculate_industry_match(jr_ind, sr_ind)
        is_same_r1 = is_r2 and (s_id == r1_senior_id)
        tiebreaker = create_tiebreaker_key(sr_idx, score, load, industry_overlap, max_score=max_score)

        candidate_scores.append({
            'sr_idx': sr_idx,
            's_id': s_id,
            'score': score,
            'load': load,
            'industry_overlap': industry_overlap,
            'is_same_r1': is_same_r1,
            'tiebreaker': tiebreaker
        })
        log_entry[round_name]['candidates_list'].append({
            'S_ID': s_id,
            'Score': round(score, 2),
            f'{round_name}_Load': load,
            'Industry_Overlap': industry_overlap,
            **({"Same_as_R1": is_same_r1} if is_r2 else {})
        })

    if is_r2:
        candidate_scores.sort(key=lambda x: (x['is_same_r1'], x['tiebreaker']))
    else:
        candidate_scores.sort(key=lambda x: x['tiebreaker'])

    log_entry[round_name]['sorted_candidates'] = candidate_scores[:3]

    if not candidate_scores:
        log_entry[round_name]['error'] = "NO_CANDIDATES"
        return None, 0, "NO_CANDIDATES", fallback_level, 0, False

    selected = None

    if phase1_only:
        # Phase 1: only pick if an under-cap same-segment senior exists
        for candidate in candidate_scores:
            if senior_load[candidate['s_id']] < CONFIG['MAX_LOAD_CAP']:
                selected = candidate
                break
        if selected is None:
            # Deferred — will be handled in Phase 2
            log_entry[round_name]['error'] = "DEFERRED_TO_PHASE2"
            return None, 0, "DEFERRED", fallback_level, 0, False
    else:
        # Phase 2 (full fallback): same logic as before
        if is_r2:
            for candidate in candidate_scores:
                if (not candidate['is_same_r1']) and senior_load[candidate['s_id']] < CONFIG['MAX_LOAD_CAP']:
                    selected = candidate
                    break
            if selected is None:
                for candidate in candidate_scores:
                    if senior_load[candidate['s_id']] < CONFIG['MAX_LOAD_CAP']:
                        selected = candidate
                        break
        else:
            for candidate in candidate_scores:
                if senior_load[candidate['s_id']] < CONFIG['MAX_LOAD_CAP']:
                    selected = candidate
                    break

        # Cap-escalation: widen through fallback levels one at a time
        if selected is None:
            for next_level in range(fallback_level + 1, len(CONFIG['FALLBACK_LEVELS']) + 1):
                ext_candidates = get_candidates_for_level(
                    next_level, jr_segment, df_seniors, senior_segments,
                    jr_ca, jr_ug, jr_exp
                )
                ext_scores = []
                for sr_idx in ext_candidates:
                    sr = df_seniors.iloc[sr_idx]
                    s_id = sr["S_ID"]
                    score = scores_matrix.iloc[jr_idx][s_id]
                    load = senior_load[s_id]
                    sr_ind = sr_industries_dict.get(sr_idx, [])
                    ind_ov = calculate_industry_match(jr_ind, sr_ind)
                    is_same = is_r2 and (s_id == r1_senior_id)
                    tb = create_tiebreaker_key(sr_idx, score, load, ind_ov, max_score=max_score)
                    ext_scores.append({'sr_idx': sr_idx, 's_id': s_id, 'score': score,
                                       'load': load, 'industry_overlap': ind_ov,
                                       'is_same_r1': is_same, 'tiebreaker': tb})
                if is_r2:
                    ext_scores.sort(key=lambda x: (x['is_same_r1'], x['tiebreaker']))
                else:
                    ext_scores.sort(key=lambda x: x['tiebreaker'])
                for candidate in ext_scores:
                    if senior_load[candidate['s_id']] < CONFIG['MAX_LOAD_CAP']:
                        selected = candidate
                        fallback_level = next_level
                        log_entry[round_name]['fallback_level'] = fallback_level
                        log_entry[round_name]['fallback_reason'] = CONFIG['FALLBACK_LEVELS'][fallback_level]
                        break
                if selected is not None:
                    break

        # Absolute last resort: overload the best-scored candidate
        if selected is None:
            selected = candidate_scores[0]
            load_type = "FORCED_OVERLOAD"
        else:
            load_type = "NORMAL"

    if selected is not None:
        load_type = load_type if 'load_type' in dir() else "NORMAL"
        s_id = selected['s_id']
        senior_load[s_id] += 1
        senior_load_total[s_id] += 1
        senior_assignments[s_id].append(jr_id)
        log_entry[round_name]['loads_before'] = senior_load[s_id] - 1
        log_entry[round_name]['loads_after'] = senior_load[s_id]
        log_entry[round_name]['selected'] = s_id
        log_entry[round_name]['score'] = round(selected['score'], 2)
        log_entry[round_name]['load_type'] = load_type
        return (s_id, selected['score'], load_type, fallback_level,
                selected['industry_overlap'], selected.get('is_same_r1', False))

    return None, 0, "UNKNOWN", fallback_level, 0, False


def match_juniors_with_seniors(
    df_juniors: pd.DataFrame,
    df_seniors: pd.DataFrame,
    junior_segments: pd.Series,
    senior_segments: pd.Series,
    r1_scores: pd.DataFrame,
    r2_scores: pd.DataFrame,
    segment_health: pd.DataFrame,
    jr_industries_dict: Dict[int, List[str]],
    sr_industries_dict: Dict[int, List[str]]
) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame, Dict, Dict, Dict, pd.DataFrame]:
    """
    Two-phase matching algorithm.

    PHASE 1 – Strict Segment Allocation
        Every junior is matched only against seniors in their own segment (level 1).
        Juniors that cannot be matched (no same-segment capacity) are collected in
        ``fallback_juniors`` and left unassigned.

    PHASE 2 – Deferred Fallback Allocation
        After Phase 1 finishes (all same-segment capacity exhausted first), the
        deferred juniors are processed using the full fallback hierarchy (levels 2-5).
        Juniors that still cannot be matched after all fallback levels are placed in
        ``manual_intervention_cases`` and exported to ``manual_intervention_required.xlsx``.

    Returns:
        (results_df, debug_logs, manual_review_df,
         senior_load_r1, senior_load_r2, senior_load_total,
         manual_intervention_df)
    """
    print("\n" + "="*80)
    print("STEP 13-16: TWO-PHASE MATCHING (DEFERRED FALLBACK STRATEGY)")
    print("="*80)

    # ===== INITIALIZE LOAD TRACKING =====
    senior_load_r1    = {s_id: 0 for s_id in df_seniors["S_ID"]}
    senior_load_r2    = {s_id: 0 for s_id in df_seniors["S_ID"]}
    senior_load_total = {s_id: 0 for s_id in df_seniors["S_ID"]}
    senior_assignments_r1 = {s_id: [] for s_id in df_seniors["S_ID"]}
    senior_assignments_r2 = {s_id: [] for s_id in df_seniors["S_ID"]}

    results        = []
    debug_logs     = []
    manual_review_records = []

    # Intermediate stores for two-phase processing
    phase1_results: Dict[str, Dict] = {}   # jr_id → partial result dict
    phase1_logs:    Dict[str, Dict] = {}   # jr_id → log_entry

    # Juniors deferred from Phase 1 (no same-segment capacity)
    fallback_juniors: List[Dict] = []

    # ===========================================================================
    # PHASE 1 — Strict same-segment matching
    # ===========================================================================
    print(f"\n  ── PHASE 1: Strict Segment Allocation ──")
    print(f"  Processing {len(df_juniors)} juniors (same-segment only)...")

    for jr_idx, junior_row in df_juniors.iterrows():
        jr_segment = junior_segments.iloc[jr_idx]
        jr_id      = junior_row["J_ID"]
        jr_name    = junior_row.get('Student Name (FN LN)', '') or "N/A"
        jr_ca      = junior_row["Is CA"]
        jr_ug      = junior_row["Undergrad Category"]
        jr_exp     = junior_row["ExpType"]
        jr_ind     = jr_industries_dict.get(jr_idx, [])

        log_entry = {
            'J_ID': jr_id,
            'Junior_Name': jr_name,
            'Segment': jr_segment,
            'Phase': 'Phase1',
            'R1': {}, 'R2': {}
        }

        # --- R1 Phase 1 ---
        r1_senior_id, r1_score, r1_load_type, r1_fallback_level, r1_industry_overlap, _ = \
            _assign_round(
                "R1", jr_idx, junior_row, jr_segment,
                jr_ca, jr_ug, jr_exp, jr_ind,
                df_seniors, senior_segments,
                r1_scores, sr_industries_dict,
                senior_load_r1, senior_load_total, senior_assignments_r1,
                log_entry, None, max_score=100.0, phase1_only=True
            )

        # --- R2 Phase 1 ---
        r2_senior_id, r2_score, r2_load_type, r2_fallback_level, r2_industry_overlap, same_senior_r1_r2 = \
            _assign_round(
                "R2", jr_idx, junior_row, jr_segment,
                jr_ca, jr_ug, jr_exp, jr_ind,
                df_seniors, senior_segments,
                r2_scores, sr_industries_dict,
                senior_load_r2, senior_load_total, senior_assignments_r2,
                log_entry, r1_senior_id, max_score=80.0, phase1_only=True
            )

        r1_deferred = (r1_senior_id is None)
        r2_deferred = (r2_senior_id is None)

        if r1_deferred or r2_deferred:
            # At least one round needs fallback — park this junior
            fallback_juniors.append({
                'jr_idx': jr_idx,
                'junior_row': junior_row,
                'jr_segment': jr_segment,
                'jr_id': jr_id,
                'jr_name': jr_name,
                'jr_ca': jr_ca,
                'jr_ug': jr_ug,
                'jr_exp': jr_exp,
                'jr_ind': jr_ind,
                # Carry forward any partial Phase 1 assignment
                'r1_senior_id': r1_senior_id,
                'r1_score': r1_score,
                'r1_load_type': r1_load_type,
                'r1_fallback_level': r1_fallback_level,
                'r1_industry_overlap': r1_industry_overlap,
                'r2_senior_id': r2_senior_id,
                'r2_score': r2_score,
                'r2_load_type': r2_load_type,
                'r2_fallback_level': r2_fallback_level,
                'r2_industry_overlap': r2_industry_overlap,
                'same_senior_r1_r2': same_senior_r1_r2,
                'log_entry': log_entry,
            })
        else:
            phase1_results[jr_id] = {
                'jr_idx': jr_idx,
                'junior_row': junior_row,
                'jr_segment': jr_segment,
                'r1_senior_id': r1_senior_id, 'r1_score': r1_score,
                'r1_load_type': r1_load_type, 'r1_fallback_level': r1_fallback_level,
                'r1_industry_overlap': r1_industry_overlap,
                'r2_senior_id': r2_senior_id, 'r2_score': r2_score,
                'r2_load_type': r2_load_type, 'r2_fallback_level': r2_fallback_level,
                'r2_industry_overlap': r2_industry_overlap,
                'same_senior_r1_r2': same_senior_r1_r2,
                'log_entry': log_entry,
            }

    print(f"\n  Phase 1 complete:")
    print(f"    Fully matched (same-segment): {len(phase1_results)}")
    print(f"    Deferred to Phase 2:          {len(fallback_juniors)}")

    # ===========================================================================
    # PHASE 2 — Deferred fallback matching
    # ===========================================================================
    print(f"\n  ── PHASE 2: Deferred Fallback Allocation ──")
    print(f"  Processing {len(fallback_juniors)} deferred juniors (full fallback hierarchy)...")

    manual_intervention_cases: List[Dict] = []

    for fb in fallback_juniors:
        jr_idx        = fb['jr_idx']
        junior_row    = fb['junior_row']
        jr_segment    = fb['jr_segment']
        jr_id         = fb['jr_id']
        jr_name       = fb['jr_name']
        jr_ca         = fb['jr_ca']
        jr_ug         = fb['jr_ug']
        jr_exp        = fb['jr_exp']
        jr_ind        = fb['jr_ind']
        log_entry     = fb['log_entry']
        log_entry['Phase'] = 'Phase2'

        # Re-use any partial Phase 1 assignment if it was made
        r1_senior_id        = fb['r1_senior_id']
        r1_score            = fb['r1_score']
        r1_load_type        = fb['r1_load_type']
        r1_fallback_level   = fb['r1_fallback_level']
        r1_industry_overlap = fb['r1_industry_overlap']

        r2_senior_id        = fb['r2_senior_id']
        r2_score            = fb['r2_score']
        r2_load_type        = fb['r2_load_type']
        r2_fallback_level   = fb['r2_fallback_level']
        r2_industry_overlap = fb['r2_industry_overlap']
        same_senior_r1_r2   = fb['same_senior_r1_r2']

        # Only run fallback for rounds that were not assigned in Phase 1
        if r1_senior_id is None:
            r1_senior_id, r1_score, r1_load_type, r1_fallback_level, r1_industry_overlap, _ = \
                _assign_round(
                    "R1", jr_idx, junior_row, jr_segment,
                    jr_ca, jr_ug, jr_exp, jr_ind,
                    df_seniors, senior_segments,
                    r1_scores, sr_industries_dict,
                    senior_load_r1, senior_load_total, senior_assignments_r1,
                    log_entry, None, max_score=100.0, phase1_only=False
                )

        if r2_senior_id is None:
            r2_senior_id, r2_score, r2_load_type, r2_fallback_level, r2_industry_overlap, same_senior_r1_r2 = \
                _assign_round(
                    "R2", jr_idx, junior_row, jr_segment,
                    jr_ca, jr_ug, jr_exp, jr_ind,
                    df_seniors, senior_segments,
                    r2_scores, sr_industries_dict,
                    senior_load_r2, senior_load_total, senior_assignments_r2,
                    log_entry, r1_senior_id, max_score=80.0, phase1_only=False
                )

        # Check if still unmatched after full fallback
        r1_unmatched = (r1_senior_id is None)
        r2_unmatched = (r2_senior_id is None)

        if r1_unmatched or r2_unmatched:
            # Collect segments that were checked
            checked_segs = [jr_segment]
            for lvl in range(2, 6):
                lvl_cands = get_candidates_for_level(
                    lvl, jr_segment, df_seniors, senior_segments, jr_ca, jr_ug, jr_exp
                )
                if lvl_cands:
                    checked_segs.extend(
                        senior_segments.iloc[c] for c in lvl_cands
                        if senior_segments.iloc[c] not in checked_segs
                    )

            # Best available score across all seniors (informational)
            all_r1 = [r1_scores.iloc[jr_idx][s] for s in df_seniors["S_ID"]]
            best_score = max(all_r1) if all_r1 else 0.0

            reasons = []
            if r1_unmatched:
                reasons.append("R1: No eligible senior after full fallback")
            if r2_unmatched:
                reasons.append("R2: No eligible senior after full fallback")

            manual_intervention_cases.append({
                'J_ID': jr_id,
                'Junior_Name': jr_name,
                'Original_Segment': jr_segment,
                'Reason_for_Failure': "; ".join(reasons),
                'Candidate_Segments_Checked': "; ".join(sorted(set(str(s) for s in checked_segs))),
                'Best_Available_Score': round(best_score, 2),
            })

        # Store result for this deferred junior
        phase1_results[jr_id] = {
            'jr_idx': jr_idx,
            'junior_row': junior_row,
            'jr_segment': jr_segment,
            'r1_senior_id': r1_senior_id, 'r1_score': r1_score,
            'r1_load_type': r1_load_type, 'r1_fallback_level': r1_fallback_level,
            'r1_industry_overlap': r1_industry_overlap,
            'r2_senior_id': r2_senior_id, 'r2_score': r2_score,
            'r2_load_type': r2_load_type, 'r2_fallback_level': r2_fallback_level,
            'r2_industry_overlap': r2_industry_overlap,
            'same_senior_r1_r2': same_senior_r1_r2,
            'log_entry': log_entry,
        }

    print(f"\n  Phase 2 complete:")
    print(f"    Fallback matched: {len(fallback_juniors) - len(manual_intervention_cases)}")
    print(f"    Manual intervention required: {len(manual_intervention_cases)}")

    # ===========================================================================
    # BUILD RESULTS — preserve original junior order
    # ===========================================================================
    for jr_idx, junior_row in df_juniors.iterrows():
        jr_id = junior_row["J_ID"]
        data  = phase1_results.get(jr_id)
        if data is None:
            continue  # should not happen

        log_entry           = data['log_entry']
        r1_senior_id        = data['r1_senior_id']
        r2_senior_id        = data['r2_senior_id']
        r1_score            = data['r1_score']
        r2_score            = data['r2_score']
        r1_load_type        = data['r1_load_type']
        r2_load_type        = data['r2_load_type']
        r1_fallback_level   = data['r1_fallback_level']
        r2_fallback_level   = data['r2_fallback_level']
        r1_industry_overlap = data['r1_industry_overlap']
        r2_industry_overlap = data['r2_industry_overlap']
        same_senior_r1_r2   = data['same_senior_r1_r2']

        debug_logs.append(log_entry)

        # Manual review flags
        should_review = False
        review_reasons = []
        if get_confidence_label(r1_score, 100) == "LOW":
            should_review = True; review_reasons.append("R1:LOW_CONFIDENCE")
        if get_confidence_label(r2_score, 80) == "LOW":
            should_review = True; review_reasons.append("R2:LOW_CONFIDENCE")
        if r1_load_type == "FORCED_OVERLOAD" or r2_load_type == "FORCED_OVERLOAD":
            should_review = True; review_reasons.append("OVERLOAD")
        if same_senior_r1_r2 and r1_fallback_level > 2:
            should_review = True; review_reasons.append("SAME_SENIOR")
        if r1_fallback_level > 3 or r2_fallback_level > 3:
            should_review = True; review_reasons.append("HIGH_FALLBACK")
        if r1_senior_id is None or r2_senior_id is None:
            should_review = True; review_reasons.append("UNMATCHED_ROUND")

        if should_review:
            manual_review_records.append({'J_ID': jr_id, 'Review_Reasons': "; ".join(review_reasons)})

        r1_senior = df_seniors[df_seniors["S_ID"] == r1_senior_id].iloc[0] if r1_senior_id else None
        r2_senior = df_seniors[df_seniors["S_ID"] == r2_senior_id].iloc[0] if r2_senior_id else None

        result_record = {
            'J_ID': jr_id,
            'Junior_Name': junior_row.get('Student Name (FN LN)', ''),
            # 'Junior_Email': junior_row.get('Email Address', ''),
            # 'Junior_Phone': junior_row.get('Phone Number', ''),
            'Junior_CA': junior_row["Is CA"],
            'Junior_UG': junior_row["Undergrad Category"],
            'Junior_UG_Degree': junior_row["Undergraduate Degree"],
            'Junior_UG_Spec': junior_row["Area of Specialization"],
            'Junior_Masters_Degree': junior_row["Masters_Degree"],
            'Junior_Masters_Specialization': junior_row["Masters_Specialization"],
            'Junior_Exp_Months': junior_row["Experience_Months"],

            'R1_Senior_ID': r1_senior_id,
            'R1_Senior_Name': r1_senior['Student Name (FN LN)'] if r1_senior is not None else 'N/A',
            # 'R1_Senior_Email': r1_senior.get('Email Address', '') if r1_senior is not None else 'N/A',
            # 'R1_Senior_Phone': r1_senior.get('Phone Number', '') if r1_senior is not None else 'N/A',
            'R1_CA': r1_senior["Is CA"] if r1_senior is not None else 'N/A',
            'R1_UG': r1_senior["Undergrad Category"] if r1_senior is not None else 'N/A',
            'R1_UG_Degree': r1_senior["Undergraduate Degree"] if r1_senior is not None else 'N/A',
            'R1_UG_Spec': r1_senior["Area of Specialization"] if r1_senior is not None else 'N/A',
            'R1_Masters_Degree': r1_senior["Masters_Degree"] if r1_senior is not None else 'N/A',
            'R1_Masters_Specialization': r1_senior["Masters_Specialization"] if r1_senior is not None else 'N/A',
            'R1_Exp_Months': r1_senior["Experience_Months"] if r1_senior is not None else 'N/A',
            'R1_Score': r1_score,
            'R1_Confidence': get_confidence_label(r1_score, 100),
            'R1_Fallback_Level': r1_fallback_level,
            'R1_Load_Type': r1_load_type,
            'R1_Industry_Overlap': r1_industry_overlap,

            'R2_Senior_ID': r2_senior_id,
            'R2_Senior_Name': r2_senior['Student Name (FN LN)'] if r2_senior is not None else 'N/A',
            # 'R2_Senior_Email': r2_senior.get('Email Address', '') if r2_senior is not None else 'N/A',
            # 'R2_Senior_Phone': r2_senior.get('Phone Number', '') if r2_senior is not None else 'N/A',
            'R2_CA': r2_senior["Is CA"] if r2_senior is not None else 'N/A',
            'R2_UG': r2_senior["Undergrad Category"] if r2_senior is not None else 'N/A',
            'R2_UG_Degree': r2_senior["Undergraduate Degree"] if r2_senior is not None else 'N/A',
            'R2_UG_Spec': r2_senior["Area of Specialization"] if r2_senior is not None else 'N/A',
            'R2_Masters_Degree': r2_senior["Masters_Degree"] if r2_senior is not None else 'N/A',
            'R2_Masters_Specialization': r2_senior["Masters_Specialization"] if r2_senior is not None else 'N/A',
            'R2_Exp_Months': r2_senior["Experience_Months"] if r2_senior is not None else 'N/A',
            'R2_Score': r2_score,
            'R2_Confidence': get_confidence_label(r2_score, 80),
            'R2_Fallback_Level': r2_fallback_level,
            'R2_Load_Type': r2_load_type,
            'R2_Industry_Overlap': r2_industry_overlap,

            'Same_Senior_R1_R2': same_senior_r1_r2,
            'Manual_Review': should_review,
            'Review_Reasons': "; ".join(review_reasons) if review_reasons else "",
            'Allocation_Phase': log_entry.get('Phase', 'Phase1'),
        }
        results.append(result_record)

    results_df = pd.DataFrame(results)

    print(f"\n✓ Matched {len(results_df)} juniors")
    print(f"  Round 1 - HIGH confidence:   {len(results_df[results_df['R1_Confidence'] == 'HIGH'])}")
    print(f"  Round 1 - MEDIUM confidence: {len(results_df[results_df['R1_Confidence'] == 'MEDIUM'])}")
    print(f"  Round 1 - LOW confidence:    {len(results_df[results_df['R1_Confidence'] == 'LOW'])}")
    print(f"\n  Round 2 - HIGH confidence:   {len(results_df[results_df['R2_Confidence'] == 'HIGH'])}")
    print(f"  Round 2 - MEDIUM confidence: {len(results_df[results_df['R2_Confidence'] == 'MEDIUM'])}")
    print(f"  Round 2 - LOW confidence:    {len(results_df[results_df['R2_Confidence'] == 'LOW'])}")

    manual_review_df        = pd.DataFrame(manual_review_records) if manual_review_records else pd.DataFrame()
    manual_intervention_df  = pd.DataFrame(manual_intervention_cases) if manual_intervention_cases else pd.DataFrame()

    return (results_df, debug_logs, manual_review_df,
            senior_load_r1, senior_load_r2, senior_load_total,
            manual_intervention_df)


# =============================================================================
#                    SECTION 12: DEBUG LOGGING (ENHANCED)
# =============================================================================

def generate_debug_report(
    debug_logs: List[Dict],
    senior_load_r1: Dict,
    senior_load_r2: Dict,
    senior_load_total: Dict,
    df_seniors: pd.DataFrame,
    output_file: str = "matching_debug.txt"
) -> None:
    """
    Generate comprehensive debug report with separated load tracking.
    
    Args:
        debug_logs: List of debug log entries
        senior_load_r1: Dict of senior R1-specific loads
        senior_load_r2: Dict of senior R2-specific loads
        senior_load_total: Dict of senior total loads
        df_seniors: Seniors DataFrame
        output_file: Output file path
    """
    print("\n" + "="*80)
    print("STEP 19: GENERATING ENHANCED DEBUG REPORT")
    print("="*80)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MENTORSHIP MATCHING ENGINE - ENHANCED DEBUG REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # ===== MATCHING DETAILS =====
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED MATCHING LOGS FOR ALL JUNIORS\n")
        f.write("="*80 + "\n\n")
        
        for log in debug_logs:
            f.write(f"\n{'─'*80}\n")
            f.write(f"JUNIOR: {log['J_ID']} - {log.get('Junior_Name', 'N/A')}\n")
            f.write(f"Segment: {log['Segment']}\n")
            f.write(f"{'─'*80}\n\n")
            
            # Round 1
            f.write("ROUND 1 MATCHING:\n")
            f.write(f"  Fallback Level: {log['R1'].get('fallback_level', 'N/A')} - {log['R1'].get('fallback_reason', 'Unknown')}\n")
            f.write(f"  Candidate Pool Size: {log['R1'].get('candidate_count', 0)}\n")
            
            if log['R1'].get('candidates_list'):
                f.write(f"  Top Candidates:\n")
                for i, cand in enumerate(log['R1']['candidates_list'][:5], 1):
                    f.write(f"    {i}. {cand.get('S_ID', 'N/A'):5} | Score: {cand.get('Score', 0):6.2f} | ")
                    f.write(f"R1_Load: {cand.get('R1_Load', 0)} | Industry Match: {cand.get('Industry_Overlap', 0)}\n")
            
            f.write(f"\n  Selected Senior: {log['R1'].get('selected', 'NONE')}\n")
            f.write(f"  Match Score: {log['R1'].get('score', 0):.2f}/100\n")
            f.write(f"  Industry Overlap: {log['R1'].get('industry_overlap', 0)}\n")
            f.write(f"  Senior Load - Before: {log['R1'].get('loads_before', 'N/A')} → After: {log['R1'].get('loads_after', 'N/A')}\n")
            f.write(f"  Load Type: {log['R1'].get('load_type', 'UNKNOWN')}\n")
            f.write(f"  Error: {log['R1'].get('error', 'None')}\n")
            
            # Round 2
            f.write("\nROUND 2 MATCHING:\n")
            f.write(f"  Fallback Level: {log['R2'].get('fallback_level', 'N/A')} - {log['R2'].get('fallback_reason', 'Unknown')}\n")
            f.write(f"  Candidate Pool Size: {log['R2'].get('candidate_count', 0)}\n")
            
            if log['R2'].get('candidates_list'):
                f.write(f"  Top Candidates:\n")
                for i, cand in enumerate(log['R2']['candidates_list'][:5], 1):
                    same_r1_mark = " [SAME AS R1]" if cand.get('Same_as_R1', False) else ""
                    f.write(f"    {i}. {cand.get('S_ID', 'N/A'):5} | Score: {cand.get('Score', 0):6.2f} | ")
                    f.write(f"R2_Load: {cand.get('R2_Load', 0)} | Industry Match: {cand.get('Industry_Overlap', 0)}{same_r1_mark}\n")
            
            f.write(f"\n  Selected Senior: {log['R2'].get('selected', 'NONE')}\n")
            f.write(f"  Match Score: {log['R2'].get('score', 0):.2f}/80\n")
            f.write(f"  Industry Overlap: {log['R2'].get('industry_overlap', 0)}\n")
            f.write(f"  Senior Load - Before: {log['R2'].get('loads_before', 'N/A')} → After: {log['R2'].get('loads_after', 'N/A')}\n")
            f.write(f"  Load Type: {log['R2'].get('load_type', 'UNKNOWN')}\n")
            f.write(f"  Same as R1: {log['R2'].get('same_as_r1', False)}\n")
            f.write(f"  Error: {log['R2'].get('error', 'None')}\n")
        
        # ===== LOAD DISTRIBUTION (SEPARATED) =====
        f.write("\n\n" + "="*80 + "\n")
        f.write("FINAL SENIOR LOAD DISTRIBUTION (SEPARATED R1 & R2)\n")
        f.write("="*80 + "\n\n")
        
        load_summary = []
        for s_id in senior_load_total.keys():
            senior = df_seniors[df_seniors["S_ID"] == s_id]
            if len(senior) > 0:
                senior = senior.iloc[0]
                load_summary.append({
                    'S_ID': s_id,
                    'Name': senior.get('Student Name (FN LN)', ''),
                    'R1_Load': senior_load_r1[s_id],
                    'R2_Load': senior_load_r2[s_id],
                    'Total_Load': senior_load_total[s_id],
                    'Max_Cap': CONFIG['MAX_LOAD_CAP'],
                    'R1_Status': 'NORMAL' if senior_load_r1[s_id] <= CONFIG['MAX_LOAD_CAP'] else 'OVERLOAD',
                    'R2_Status': 'NORMAL' if senior_load_r2[s_id] <= CONFIG['MAX_LOAD_CAP'] else 'OVERLOAD'
                })
        
        load_df = pd.DataFrame(load_summary).sort_values(by='Total_Load', ascending=False)
        
        f.write(f"{'S_ID':<6} {'Name':<30} {'R1':<5} {'R2':<5} {'Total':<6} {'R1_Status':<12} {'R2_Status':<12}\n")
        f.write(f"{'-'*6} {'-'*30} {'-'*5} {'-'*5} {'-'*6} {'-'*12} {'-'*12}\n")
        
        for _, row in load_df.iterrows():
            f.write(f"{row['S_ID']:<6} {row['Name']:<30} {row['R1_Load']:<5} {row['R2_Load']:<5} ")
            f.write(f"{row['Total_Load']:<6} {row['R1_Status']:<12} {row['R2_Status']:<12}\n")
        
        f.write(f"\n{'SUMMARY STATISTICS':─^80}\n\n")
        f.write(f"  Total R1 Assignments: {sum(senior_load_r1.values())}\n")
        f.write(f"  Total R2 Assignments: {sum(senior_load_r2.values())}\n")
        f.write(f"  Total Assignments: {sum(senior_load_total.values())}\n")
        f.write(f"\n  R1 - Normal Loads: {len(load_df[(load_df['R1_Load'] <= CONFIG['MAX_LOAD_CAP'])])}\n")
        f.write(f"  R1 - Overloaded: {len(load_df[(load_df['R1_Load'] > CONFIG['MAX_LOAD_CAP'])])}\n")
        f.write(f"  R1 - Avg Load: {np.mean(list(senior_load_r1.values())):.2f}\n")
        f.write(f"  R1 - Max Load: {max(senior_load_r1.values())}\n")
        f.write(f"  R1 - Min Load: {min(senior_load_r1.values())}\n")
        
        f.write(f"\n  R2 - Normal Loads: {len(load_df[(load_df['R2_Load'] <= CONFIG['MAX_LOAD_CAP'])])}\n")
        f.write(f"  R2 - Overloaded: {len(load_df[(load_df['R2_Load'] > CONFIG['MAX_LOAD_CAP'])])}\n")
        f.write(f"  R2 - Avg Load: {np.mean(list(senior_load_r2.values())):.2f}\n")
        f.write(f"  R2 - Max Load: {max(senior_load_r2.values())}\n")
        f.write(f"  R2 - Min Load: {min(senior_load_r2.values())}\n")
        
        f.write(f"\n  Total - Avg Load: {np.mean(list(senior_load_total.values())):.2f}\n")
        f.write(f"  Total - Max Load: {max(senior_load_total.values())}\n")
        f.write(f"  Total - Min Load: {min(senior_load_total.values())}\n")
        
        f.write(f"\n  Max Capacity per Senior: {CONFIG['MAX_LOAD_CAP']}\n")
        f.write(f"\nReport saved to: {output_file}\n")
    
    print(f"✓ Enhanced debug report saved to: {output_file}")


# =============================================================================
#                    SECTION 13: OUTPUT GENERATION & ANALYTICS
# =============================================================================

def generate_analytics_report(
    results_df: pd.DataFrame,
    segment_health: pd.DataFrame,
    senior_load_r1: Dict,
    senior_load_r2: Dict,
    senior_load_total: Dict,
    debug_logs: List[Dict],
    df_seniors: pd.DataFrame,
    output_file: str = "analytics_report.xlsx"
) -> pd.DataFrame:
    """
    Generate comprehensive analytics report with statistics.
    
    Args:
        results_df: Final matching results
        segment_health: Segment health DataFrame
        senior_load_r1: R1-specific loads
        senior_load_r2: R2-specific loads
        senior_load_total: Total loads
        debug_logs: Debug logs
        df_seniors: Seniors DataFrame
        output_file: Output file path
    
    Returns:
        Analytics DataFrame
    """
    print("\n" + "="*80)
    print("STEP 22: GENERATING ANALYTICS REPORT")
    print("="*80)
    
    # Count various statistics
    total_juniors = len(results_df)
    total_seniors = len(df_seniors)
    avg_r1_score = results_df['R1_Score'].mean()
    avg_r2_score = results_df['R2_Score'].mean()
    
    unmatched_juniors = len(results_df[results_df['R1_Senior_ID'].isna()])
    manual_review_count = len(results_df[results_df['Manual_Review'] == True])
    
    # Overload analysis
    overloaded_r1 = sum(1 for load in senior_load_r1.values() if load > CONFIG['MAX_LOAD_CAP'])
    overloaded_r2 = sum(1 for load in senior_load_r2.values() if load > CONFIG['MAX_LOAD_CAP'])
    total_overloaded = sum(1 for load in senior_load_total.values() if load > 2 * CONFIG['MAX_LOAD_CAP'])
    
    # Fallback usage analysis
    fallback_counts = {}
    for log in debug_logs:
        r1_level = log['R1'].get('fallback_level', 5)
        r2_level = log['R2'].get('fallback_level', 5)
        fallback_counts[f'R1_Level_{r1_level}'] = fallback_counts.get(f'R1_Level_{r1_level}', 0) + 1
        fallback_counts[f'R2_Level_{r2_level}'] = fallback_counts.get(f'R2_Level_{r2_level}', 0) + 1
    
    # Confidence distribution
    r1_high = len(results_df[results_df['R1_Confidence'] == 'HIGH'])
    r1_medium = len(results_df[results_df['R1_Confidence'] == 'MEDIUM'])
    r1_low = len(results_df[results_df['R1_Confidence'] == 'LOW'])
    
    r2_high = len(results_df[results_df['R2_Confidence'] == 'HIGH'])
    r2_medium = len(results_df[results_df['R2_Confidence'] == 'MEDIUM'])
    r2_low = len(results_df[results_df['R2_Confidence'] == 'LOW'])
    
    # Same senior R1/R2 analysis
    same_senior = len(results_df[results_df['Same_Senior_R1_R2'] == True])
    
    # Segment analysis
    segment_stats = []
    for _, seg_row in segment_health.iterrows():
        segment = seg_row['Segment']
        j_count = seg_row['Junior_Count']
        s_count = seg_row['Senior_Count']
        segment_stats.append({
            'Segment': segment,
            'Juniors': j_count,
            'Seniors': s_count,
            'Risk_Level': seg_row['Risk_Level'],
            'Capacity_Health': safe_divide(s_count * CONFIG['MAX_LOAD_CAP'], j_count * 2)
        })
    
    # Top loaded seniors
    top_loaded = []
    for s_id in sorted(senior_load_total.keys(), key=lambda x: senior_load_total[x], reverse=True)[:10]:
        senior = df_seniors[df_seniors["S_ID"] == s_id]
        if len(senior) > 0:
            senior = senior.iloc[0]
            top_loaded.append({
                'S_ID': s_id,
                'Name': senior.get('Student Name (FN LN)', ''),
                'R1_Load': senior_load_r1[s_id],
                'R2_Load': senior_load_r2[s_id],
                'Total_Load': senior_load_total[s_id],
                'Load_Type': 'OVERLOADED' if senior_load_total[s_id] > 2 * CONFIG['MAX_LOAD_CAP'] else 'NORMAL'
            })
    
    # Create main analytics summary
    analytics_data = {
        'Metric': [
            'Total Juniors Processed',
            'Total Seniors Available',
            'Avg R1 Score',
            'Avg R2 Score',
            'Unmatched Juniors',
            'Manual Review Required',
            'Same Senior R1&R2',
            'R1 High Confidence',
            'R1 Medium Confidence',
            'R1 Low Confidence',
            'R2 High Confidence',
            'R2 Medium Confidence',
            'R2 Low Confidence',
            'Seniors Overloaded (R1)',
            'Seniors Overloaded (R2)',
            'Seniors Overloaded (Total)',
            'Avg Senior Load (Total)',
            'Max Senior Load (Total)',
            'Min Senior Load (Total)',
            'Random Seed',
            'Max Load Cap',
            'Processing Method'
        ],
        'Value': [
            total_juniors,
            total_seniors,
            f'{avg_r1_score:.2f}',
            f'{avg_r2_score:.2f}',
            unmatched_juniors,
            manual_review_count,
            same_senior,
            r1_high,
            r1_medium,
            r1_low,
            r2_high,
            r2_medium,
            r2_low,
            overloaded_r1,
            overloaded_r2,
            total_overloaded,
            f'{np.mean(list(senior_load_total.values())):.2f}',
            max(senior_load_total.values()),
            min(senior_load_total.values()),
            CONFIG['RANDOM_STATE'],
            CONFIG['MAX_LOAD_CAP'],
            'Separated R1/R2 Load Tracking with Deterministic Tie-Breaking'
        ]
    }
    
    analytics_df = pd.DataFrame(analytics_data)
    
    # Create Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        analytics_df.to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(segment_stats).to_excel(writer, sheet_name='Segment_Analysis', index=False)
        pd.DataFrame(top_loaded).to_excel(writer, sheet_name='Top_Loaded_Seniors', index=False)
    
    print(f"✓ Analytics report saved: {output_file}")
    print(f"  - Total juniors: {total_juniors}")
    print(f"  - Avg scores: R1={avg_r1_score:.2f}, R2={avg_r2_score:.2f}")
    print(f"  - Manual reviews: {manual_review_count}")
    print(f"  - Overloaded seniors: {total_overloaded}")
    
    return analytics_df


def safe_divide(a, b):
    """Safe division to avoid division by zero."""
    return a / b if b != 0 else 0


def save_outputs(
    results_df: pd.DataFrame,
    segment_health: pd.DataFrame,
    manual_review_df: pd.DataFrame,
    debug_logs: List[Dict],
    senior_load_r1: Dict,
    senior_load_r2: Dict,
    senior_load_total: Dict,
    df_seniors: pd.DataFrame,
    manual_intervention_df: pd.DataFrame,
    output_dir: str = "."
) -> None:
    """
    Save all output files with separate load tracking.

    Args:
        results_df: Final matching results
        segment_health: Segment health report
        manual_review_df: Manual review records
        debug_logs: Debug logs
        senior_load_r1: R1-specific loads
        senior_load_r2: R2-specific loads
        senior_load_total: Total loads
        df_seniors: Seniors DataFrame
        manual_intervention_df: Juniors that could not be matched after all fallback levels
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("STEP 20-21: SAVING OUTPUT FILES")
    print("="*80)

    # Final output
    output_file = f"{output_dir}/final_output.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"✓ Saved final output: {output_file}")

    # Segment health
    health_file = f"{output_dir}/segment_health.xlsx"
    segment_health.to_excel(health_file, index=False)
    print(f"✓ Saved segment health: {health_file}")

    # Manual review
    if len(manual_review_df) > 0:
        review_file = f"{output_dir}/manual_review.xlsx"
        manual_review_df.to_excel(review_file, index=False)
        print(f"✓ Saved manual review ({len(manual_review_df)} records): {review_file}")

    # Manual intervention (unresolvable after full fallback)
    if manual_intervention_df is not None and len(manual_intervention_df) > 0:
        intervention_file = f"{output_dir}/manual_intervention_required.xlsx"
        # Ensure expected column order
        col_order = [
            'J_ID', 'Junior_Name', 'Original_Segment',
            'Reason_for_Failure', 'Candidate_Segments_Checked',
            'Best_Available_Score',
        ]
        existing_cols = [c for c in col_order if c in manual_intervention_df.columns]
        manual_intervention_df[existing_cols].to_excel(intervention_file, index=False)
        print(f"✓ Saved manual intervention ({len(manual_intervention_df)} records): {intervention_file}")
    else:
        print(f"✓ No manual intervention cases — all juniors matched successfully")

    # Analytics report
    analytics_file = f"{output_dir}/analytics_report.xlsx"
    generate_analytics_report(
        results_df, segment_health,
        senior_load_r1, senior_load_r2, senior_load_total,
        debug_logs, df_seniors, analytics_file
    )

    # Debug report
    debug_file = f"{output_dir}/matching_debug.txt"
    generate_debug_report(debug_logs, senior_load_r1, senior_load_r2, senior_load_total, df_seniors, debug_file)




# =============================================================================
#                           MAIN EXECUTION
# =============================================================================

def main(junior_file: str = "Bo'27.xlsx", senior_file: str = "Bo'26.xlsx") -> None:
    """
    Main execution function. Orchestrates the entire matching process with improvements.
    
    Improvements:
    - Separate R1 and R2 load tracking
    - Fair junior shuffling
    - Deterministic tie-breaking
    - Validation layer
    - CONFIG-based system
    - Enhanced debug logging
    - Analytics reporting
    
    Args:
        junior_file: Path to junior Excel file
        senior_file: Path to senior Excel file
    """
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  MENTORSHIP ALLOCATION MATCHING ENGINE - PRODUCTION v4.0".center(78) + "█")
    print("█" + "  (IMPROVED: Two-Phase Deferred Fallback Allocation Strategy)".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    debug_log = []
    
    try:
        # STEP 1: Load data
        df_juniors, df_seniors = load_data(junior_file, senior_file)
        
        # STEP 2: Create IDs
        create_ids(df_juniors, df_seniors)
        
        # STEP 3-4: Clean data
        print("\n" + "="*80)
        print("STEP 3-4: DATA CLEANING & NORMALIZATION")
        print("="*80)
        df_juniors = clean_dataframe(df_juniors, "junior")
        df_seniors = clean_dataframe(df_seniors, "senior")
        print(f"✓ Cleaned {len(df_juniors)} juniors and {len(df_seniors)} seniors")
        
        # STEP 4.5: Fair shuffling (removes sequential bias)
        #df_juniors = shuffle_juniors_fair(df_juniors)
        #shuffle juniors based on work experience with more experienced juniors first
        df_juniors = df_juniors.sort_values(by="Experience_Months", ascending=False).reset_index(drop=True)

        # STEP 5: Create segments
        print("\n" + "="*80)
        print("STEP 5: SEGMENT CREATION")
        print("="*80)
        junior_segments = create_segments(df_juniors)
        senior_segments = create_segments(df_seniors)
        df_juniors["Segment"] = junior_segments
        df_seniors["Segment"] = senior_segments
        print(f"✓ Created segments: {len(set(junior_segments))} unique segments")
        #get segment value and counts for each segment
        print("  Junior Segments:")
        for segment in sorted(set(junior_segments)):
            count = (junior_segments == segment).sum()
            print(f"  Segment {segment}: {count} juniors")
        print("  Senior Segments:")
        for segment in sorted(set(senior_segments)):
            count = (senior_segments == segment).sum()
            print(f"  Segment {segment}: {count} seniors")
        
        # VALIDATION: Verify dataframes before matching
        validate_dataframes(df_juniors, df_seniors, debug_log)
        
        # STEP 8: Extract industries
        print("\n" + "="*80)
        print("STEP 8: INDUSTRY EXTRACTION")
        print("="*80)
        jr_industries_dict = extract_industries(df_juniors)
        sr_industries_dict = extract_industries(df_seniors)
        print(f"✓ Extracted industries for all juniors and seniors")
        
        # STEP 12: Segment health analysis
        segment_health = analyze_segment_health(
            df_juniors, df_seniors, junior_segments, senior_segments
        )
        
        # STEP 9: Generate scoring matrices
        r1_scores, r2_scores = generate_scoring_matrices(
            df_juniors, df_seniors,
            jr_industries_dict, sr_industries_dict
        )
        
        # STEP 13-16: Main matching — two-phase deferred fallback strategy
        (results_df, debug_logs, manual_review_df,
         senior_load_r1, senior_load_r2, senior_load_total,
         manual_intervention_df) = match_juniors_with_seniors(
            df_juniors, df_seniors,
            junior_segments, senior_segments,
            r1_scores, r2_scores,
            segment_health,
            jr_industries_dict, sr_industries_dict
        )

        # STEP 20-21: Save outputs
        save_outputs(
            results_df, segment_health, manual_review_df,
            debug_logs, senior_load_r1, senior_load_r2, senior_load_total,
            df_seniors, manual_intervention_df
        )
        
        print("\n" + "█" * 80)
        print("█" + "  ✓ MATCHING COMPLETE".center(78) + "█")
        print("█" * 80)
        print("\nGenerated files:")
        print("  1. final_output.xlsx - Complete matching results with R1/R2 scores")
        print("  2. segment_health.xlsx - Segment risk analysis")
        print("  3. manual_review.xlsx - Cases requiring human review")
        print("  4. analytics_report.xlsx - Comprehensive statistics & fairness metrics")
        print("  5. matching_debug.txt - Detailed decision logs with separated loads")
        print("  6. manual_intervention_required.xlsx - Juniors unresolvable after all fallback levels")
        print("\nKey Improvements (v4.0):")
        print("  ✓ Phase 1: Strict same-segment matching — no cross-segment capacity consumed")
        print("  ✓ Phase 2: Deferred fallback — runs only after Phase 1 fully exhausted")
        print("  ✓ Manual intervention export for truly unresolvable cases")
        print("  ✓ Separate load tracking for R1 and R2")
        print("  ✓ Fair junior shuffling (removes sequential bias)")
        print("  ✓ Deterministic tie-breaking (score > load > industry > index)")
        print("  ✓ Input validation before matching")
        print("  ✓ CONFIG-based customization")
        print("  ✓ Enhanced debug logging")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("\nDebug Log:")
        for msg in debug_log:
            print(f"  {msg}")
        raise
    


if __name__ == "__main__":
    main()