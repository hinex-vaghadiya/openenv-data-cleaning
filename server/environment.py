"""
Data Cleaning Environment - Core Logic

Simulates real-world data cleaning tasks where an AI agent must identify
and fix data quality issues in messy tabular datasets.

Three tasks with increasing difficulty:
  - Task 1 (Easy): Basic cleaning   duplicates, missing values, simple type issues
  - Task 2 (Medium): Format standardization  dates, phone numbers, addresses, categories
  - Task 3 (Hard): Complex multi-issue dataset  outliers, typos, mixed formats, schema issues
"""

import copy
import math
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from openenv.core.env_server.interfaces import Environment


# =============================================================================
# Dataset Generators  create messy datasets for each task
# =============================================================================

def _generate_easy_dataset(seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict]:
    """Task 1 (Easy): Employee records with basic data quality issues.

    Issues introduced:
      - Duplicate rows
      - Missing values in 'email' and 'salary'
      - Salary stored as string instead of number
      - Some null department entries
    
    Returns: (dirty_data, clean_data, issue_manifest)
    """
    rng = random.Random(seed)
    
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    clean_rows = []
    for i in range(30):
        fn = rng.choice(first_names)
        ln = rng.choice(last_names)
        dept = rng.choice(departments)
        salary = round(rng.uniform(40000, 120000), 2)
        clean_rows.append({
            "id": i + 1,
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}@company.com",
            "department": dept,
            "salary": salary,
            "active": rng.choice([True, False]),
        })
    
    # Create dirty version
    dirty_rows = copy.deepcopy(clean_rows)
    
    # 1. Add 5 duplicate rows
    for _ in range(5):
        dirty_rows.append(copy.deepcopy(rng.choice(dirty_rows[:30])))
    
    # 2. Set some emails to None (missing)
    for idx in rng.sample(range(30), 4):
        dirty_rows[idx]["email"] = None
    
    # 3. Set some salaries to string format
    for idx in rng.sample(range(30), 5):
        dirty_rows[idx]["salary"] = str(dirty_rows[idx]["salary"])
    
    # 4. Set some departments to None
    for idx in rng.sample(range(30), 3):
        dirty_rows[idx]["department"] = None
    
    # 5. Set some salaries to None  
    for idx in rng.sample(range(30), 3):
        dirty_rows[idx]["salary"] = None
    
    rng.shuffle(dirty_rows)
    
    issues = {
        "duplicates": 5,
        "missing_emails": 4,
        "missing_departments": 3,
        "missing_salaries": 3,
        "string_salaries": 5,
        "total_issues": 20,
    }
    
    return dirty_rows, clean_rows, issues


def _generate_medium_dataset(seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict]:
    """Task 2 (Medium): Customer records with format standardization issues.

    Issues introduced:
      - Inconsistent date formats (MM/DD/YYYY, DD-MM-YYYY, YYYY.MM.DD)
      - Phone numbers in various formats
      - Inconsistent state abbreviations vs full names
      - Category typos/inconsistencies  
      - Duplicates
      - Missing values
    """
    rng = random.Random(seed)
    
    first_names = ["Michael", "Sarah", "James", "Emily", "Robert", "Jessica", "David", 
                   "Ashley", "William", "Jennifer", "Daniel", "Amanda", "Christopher", "Stephanie",
                   "Matthew", "Nicole", "Andrew", "Elizabeth", "Joshua", "Megan"]
    last_names = ["Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson",
                  "Moore", "Allen", "Young", "King", "Wright", "Lopez", "Hill", "Scott",
                  "Green", "Adams", "Baker", "Gonzalez", "Nelson"]
    
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    states_full = ["New York", "California", "Illinois", "Texas", "Arizona"]
    states_abbr = ["NY", "CA", "IL", "TX", "AZ"]
    categories = ["Premium", "Standard", "Basic"]
    
    clean_rows = []
    for i in range(40):
        fn = rng.choice(first_names)
        ln = rng.choice(last_names)
        city_idx = rng.randint(0, 4)
        phone = f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
        signup_date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        
        clean_rows.append({
            "customer_id": f"CUST-{i+1:04d}",
            "full_name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}{rng.randint(1,99)}@email.com",
            "phone": phone,
            "city": cities[city_idx],
            "state": states_abbr[city_idx],
            "signup_date": signup_date,
            "plan": rng.choice(categories),
            "monthly_spend": round(rng.uniform(9.99, 299.99), 2),
        })
    
    dirty_rows = copy.deepcopy(clean_rows)
    
    # 1. Mess up date formats
    date_formats_dirty = [
        lambda d: f"{d[5:7]}/{d[8:10]}/{d[:4]}",        # MM/DD/YYYY
        lambda d: f"{d[8:10]}-{d[5:7]}-{d[:4]}",        # DD-MM-YYYY
        lambda d: f"{d[:4]}.{d[5:7]}.{d[8:10]}",        # YYYY.MM.DD
    ]
    for idx in range(40):
        fmt = rng.choice(date_formats_dirty)
        dirty_rows[idx]["signup_date"] = fmt(dirty_rows[idx]["signup_date"])
    
    # 2. Mess up phone formats
    phone_formats = [
        lambda p: p.replace("(", "").replace(")", "").replace(" ", "").replace("-", ""),  # plain digits
        lambda p: p.replace("(", "").replace(")", "-"),  # dashes only
        lambda p: p,  # keep original
    ]
    for idx in range(40):
        fmt = rng.choice(phone_formats)
        dirty_rows[idx]["phone"] = fmt(dirty_rows[idx]["phone"])
    
    # 3. Mix state abbreviations and full names
    for idx in rng.sample(range(40), 15):
        state_abbr = dirty_rows[idx]["state"]
        state_idx = states_abbr.index(state_abbr) if state_abbr in states_abbr else 0
        dirty_rows[idx]["state"] = states_full[state_idx]
    
    # 4. Category typos
    typo_map = {"Premium": ["premium", "Preminum", "PREMIUM", "Prem"],
                "Standard": ["standard", "Standrd", "STANDARD", "Std"],
                "Basic": ["basic", "Basci", "BASIC", "Bsc"]}
    for idx in rng.sample(range(40), 12):
        plan = dirty_rows[idx]["plan"]
        dirty_rows[idx]["plan"] = rng.choice(typo_map.get(plan, [plan]))
    
    # 5. Add 4 duplicates
    for _ in range(4):
        dirty_rows.append(copy.deepcopy(rng.choice(dirty_rows[:40])))
    
    # 6. Missing values
    for idx in rng.sample(range(40), 5):
        dirty_rows[idx]["email"] = None
    for idx in rng.sample(range(40), 3):
        dirty_rows[idx]["phone"] = None
    for idx in rng.sample(range(40), 4):
        dirty_rows[idx]["monthly_spend"] = None
    
    rng.shuffle(dirty_rows)
    
    issues = {
        "inconsistent_dates": 40,
        "inconsistent_phones": 40,
        "inconsistent_states": 15,
        "category_typos": 12,
        "duplicates": 4,
        "missing_values": 12,
        "total_issues": 123,
    }
    
    return dirty_rows, clean_rows, issues


def _generate_hard_dataset(seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict]:
    """Task 3 (Hard): Sales transaction dataset with complex multi-dimensional issues.

    Issues introduced:
      - All issues from easy and medium tasks
      - Statistical outliers in revenue
      - Negative quantities (invalid)
      - Future dates (impossible sales)
      - Cross-field inconsistencies (city-state mismatch)
      - Redundant/irrelevant columns
      - Mixed encoding issues in product names
      - Schema issues (wrong column names)
    """
    rng = random.Random(seed)
    
    products = ["Widget A", "Widget B", "Gadget Pro", "Super Tool", "Basic Kit",
                "Premium Pack", "Starter Set", "Advanced Module", "Elite Bundle", "Economy Pack"]
    regions = ["North", "South", "East", "West"]
    sales_reps = ["Rep_001", "Rep_002", "Rep_003", "Rep_004", "Rep_005"]
    
    clean_rows = []
    for i in range(60):
        qty = rng.randint(1, 100)
        unit_price = round(rng.uniform(10.0, 500.0), 2)
        revenue = round(qty * unit_price, 2)
        
        clean_rows.append({
            "transaction_id": f"TXN-{i+1:05d}",
            "date": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "product": rng.choice(products),
            "quantity": qty,
            "unit_price": unit_price,
            "revenue": revenue,
            "region": rng.choice(regions),
            "sales_rep": rng.choice(sales_reps),
            "status": rng.choice(["completed", "pending", "shipped"]),
        })
    
    dirty_rows = copy.deepcopy(clean_rows)
    
    # 1. Add outliers in revenue (impossibly high/low)
    for idx in rng.sample(range(60), 5):
        dirty_rows[idx]["revenue"] = round(rng.uniform(500000, 1000000), 2)
    
    # 2. Negative quantities
    for idx in rng.sample(range(60), 4):
        dirty_rows[idx]["quantity"] = -rng.randint(1, 50)
    
    # 3. Future dates
    for idx in rng.sample(range(60), 5):
        dirty_rows[idx]["date"] = f"2027-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    
    # 4. Mixed date formats
    for idx in rng.sample(range(60), 15):
        d = dirty_rows[idx]["date"]
        fmt = rng.choice([
            lambda x: f"{x[5:7]}/{x[8:10]}/{x[:4]}",
            lambda x: f"{x[8:10]}-{x[5:7]}-{x[:4]}",
        ])
        dirty_rows[idx]["date"] = fmt(d)
    
    # 5. Product name encoding issues
    encoding_mess = {"Widget A": "Widget\u00a0A", "Gadget Pro": "Gadget\u00a0Pro", 
                     "Super Tool": "Super  Tool", "Basic Kit": "basic kit"}
    for idx in rng.sample(range(60), 10):
        prod = dirty_rows[idx]["product"]
        if prod in encoding_mess:
            dirty_rows[idx]["product"] = encoding_mess[prod]
    
    # 6. Status inconsistencies
    status_typos = {"completed": ["Completed", "COMPLETED", "complete", "Compelted"],
                    "pending": ["Pending", "PENDING", "pendign"],
                    "shipped": ["Shipped", "SHIPPED", "shiped"]}
    for idx in rng.sample(range(60), 15):
        s = dirty_rows[idx]["status"]
        if s in status_typos:
            dirty_rows[idx]["status"] = rng.choice(status_typos[s])
    
    # 7. Add irrelevant columns
    for row in dirty_rows:
        row["_internal_id"] = rng.randint(10000, 99999)
        row["debug_flag"] = rng.choice(["Y", "N", None])
        row["legacy_code"] = f"LC-{rng.randint(1,999):03d}"
    
    # 8. Revenue stored as string sometimes
    for idx in rng.sample(range(60), 8):
        if dirty_rows[idx]["revenue"] is not None:
            dirty_rows[idx]["revenue"] = f"${dirty_rows[idx]['revenue']}"
    
    # 9. Add duplicates
    for _ in range(6):
        dirty_rows.append(copy.deepcopy(rng.choice(dirty_rows[:60])))
    
    # 10. Missing values
    for idx in rng.sample(range(60), 6):
        dirty_rows[idx]["revenue"] = None
    for idx in rng.sample(range(60), 4):
        dirty_rows[idx]["sales_rep"] = None
    for idx in rng.sample(range(60), 3):
        dirty_rows[idx]["region"] = None
    
    # 11. Rename a column to wrong name in some rows (schema inconsistency)
    for idx in rng.sample(range(60), 5):
        dirty_rows[idx]["unit_cost"] = dirty_rows[idx].pop("unit_price")
    
    rng.shuffle(dirty_rows)
    
    issues = {
        "outlier_revenue": 5,
        "negative_quantities": 4,
        "future_dates": 5,
        "inconsistent_dates": 15,
        "encoding_issues": 10,
        "status_typos": 15,
        "irrelevant_columns": 3,
        "string_revenue": 8,
        "duplicates": 6,
        "missing_values": 13,
        "schema_inconsistency": 5,
        "total_issues": 89,
    }
    
    return dirty_rows, clean_rows, issues


# =============================================================================
# Grading Logic
# =============================================================================

def _compute_quality_score(current_data: List[Dict], clean_data: List[Dict], 
                           task_id: str, issues: Dict) -> float:
    """Compute a data quality score between 0.0 and 1.0.
    
    Score is computed across multiple dimensions:
      - Correct row count (no extra duplicates, no missing rows)
      - Missing value reduction
      - Data type correctness
      - Format consistency
      - Outlier handling
      - Schema correctness
    """
    score = 0.0
    weights = {}
    
    if task_id == "task_easy":
        weights = {"row_count": 0.25, "missing_reduced": 0.30, "type_correct": 0.25, "no_duplicates": 0.20}
    elif task_id == "task_medium":
        weights = {"row_count": 0.15, "missing_reduced": 0.20, "format_consistent": 0.30, 
                   "no_duplicates": 0.15, "typos_fixed": 0.20}
    elif task_id == "task_hard":
        weights = {"row_count": 0.10, "missing_reduced": 0.15, "format_consistent": 0.15,
                   "no_duplicates": 0.10, "outliers_fixed": 0.15, "schema_correct": 0.15,
                   "type_correct": 0.10, "irrelevant_removed": 0.10}
    
    clean_row_count = len(clean_data)
    current_row_count = len(current_data)
    
    # 1. Row count accuracy
    if "row_count" in weights:
        if clean_row_count > 0:
            ratio = min(current_row_count, clean_row_count) / max(current_row_count, clean_row_count)
            score += weights["row_count"] * ratio
    
    # 2. Missing value reduction
    if "missing_reduced" in weights:
        total_cells = 0
        missing_cells = 0
        for row in current_data:
            for v in row.values():
                total_cells += 1
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    missing_cells += 1
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            score += weights["missing_reduced"] * max(0, 1.0 - missing_ratio * 10)
    
    # 3. No duplicates
    if "no_duplicates" in weights:
        seen = set()
        dup_count = 0
        for row in current_data:
            key_fields = {k: v for k, v in sorted(row.items()) if k not in ("_internal_id", "debug_flag", "legacy_code")}
            key = str(key_fields)
            if key in seen:
                dup_count += 1
            seen.add(key)
        if current_row_count > 0:
            dup_ratio = dup_count / current_row_count
            score += weights["no_duplicates"] * max(0, 1.0 - dup_ratio * 5)
    
    # 4. Type correctness (e.g., salary should be numeric, not string)
    if "type_correct" in weights:
        type_errors = 0
        total_checks = 0
        numeric_cols = {"salary", "revenue", "unit_price", "quantity", "monthly_spend"}
        for row in current_data:
            for col, val in row.items():
                if col in numeric_cols and val is not None:
                    total_checks += 1
                    if isinstance(val, str):
                        type_errors += 1
        if total_checks > 0:
            score += weights["type_correct"] * (1.0 - type_errors / total_checks)
        else:
            score += weights["type_correct"]
    
    # 5. Format consistency (dates, phones, categories)
    if "format_consistent" in weights:
        format_score = 1.0
        # Check dates
        date_cols = [col for col in (["signup_date", "date"]) if any(col in row for row in current_data)]
        if date_cols:
            consistent_dates = 0
            total_dates = 0
            for row in current_data:
                for dc in date_cols:
                    if dc in row and row[dc] is not None:
                        total_dates += 1
                        val = str(row[dc])
                        # Check YYYY-MM-DD format
                        if len(val) == 10 and val[4] == "-" and val[7] == "-":
                            consistent_dates += 1
            if total_dates > 0:
                format_score = consistent_dates / total_dates
        score += weights["format_consistent"] * format_score
    
    # 6. Typos fixed
    if "typos_fixed" in weights:
        valid_values = {
            "plan": {"Premium", "Standard", "Basic"},
            "status": {"completed", "pending", "shipped"},
        }
        correct = 0
        total = 0
        for row in current_data:
            for col, valid_set in valid_values.items():
                if col in row and row[col] is not None:
                    total += 1
                    if row[col] in valid_set:
                        correct += 1
        if total > 0:
            score += weights["typos_fixed"] * (correct / total)
        else:
            score += weights["typos_fixed"]
    
    # 7. Outliers fixed
    if "outliers_fixed" in weights:
        outlier_count = 0
        total_numeric = 0
        for row in current_data:
            if "revenue" in row and row["revenue"] is not None:
                try:
                    val = float(str(row["revenue"]).replace("$", "").replace(",", ""))
                    total_numeric += 1
                    if val > 100000 or val < 0:
                        outlier_count += 1
                except (ValueError, TypeError):
                    pass
            if "quantity" in row and row["quantity"] is not None:
                try:
                    val = int(row["quantity"])
                    total_numeric += 1
                    if val < 0:
                        outlier_count += 1
                except (ValueError, TypeError):
                    pass
        if total_numeric > 0:
            score += weights["outliers_fixed"] * max(0, 1.0 - outlier_count / total_numeric * 5)
        else:
            score += weights["outliers_fixed"]
    
    # 8. Schema correct
    if "schema_correct" in weights:
        expected_cols = set()
        if clean_data:
            expected_cols = set(clean_data[0].keys())
        
        schema_correct = 0
        for row in current_data:
            row_cols = set(row.keys())
            if row_cols == expected_cols:
                schema_correct += 1
        if current_row_count > 0:
            score += weights["schema_correct"] * (schema_correct / current_row_count)
    
    # 9. Irrelevant columns removed
    if "irrelevant_removed" in weights:
        irrelevant = {"_internal_id", "debug_flag", "legacy_code"}
        has_irrelevant = False
        for row in current_data:
            if any(col in row for col in irrelevant):
                has_irrelevant = True
                break
        score += weights["irrelevant_removed"] * (0.0 if has_irrelevant else 1.0)
    
    return round(min(max(score, 0.0), 1.0), 4)


# =============================================================================
# Environment Implementation
# =============================================================================

TASK_CONFIGS = {
    "task_easy": {
        "name": "Basic Data Cleaning",
        "description": (
            "Clean an employee dataset with basic issues: duplicate rows, missing values "
            "in email/salary/department columns, and salary values stored as strings instead of numbers. "
            "Goal: Remove duplicates, fill or handle missing values, and ensure correct data types."
        ),
        "generator": _generate_easy_dataset,
        "max_steps": 15,
        "difficulty": "easy",
    },
    "task_medium": {
        "name": "Format Standardization",
        "description": (
            "Clean a customer dataset with format inconsistencies: dates in mixed formats "
            "(MM/DD/YYYY, DD-MM-YYYY, YYYY.MM.DD), phone numbers in various formats, "
            "inconsistent state abbreviations vs full names, and typos in plan categories. "
            "Also handle duplicates and missing values. Goal: Standardize all formats."
        ),
        "generator": _generate_medium_dataset,
        "max_steps": 25,
        "difficulty": "medium",
    },
    "task_hard": {
        "name": "Complex Multi-Issue Cleaning",
        "description": (
            "Clean a sales transaction dataset with complex, interleaved issues: "
            "statistical outliers in revenue, negative quantities, future dates, "
            "mixed date formats, encoding issues in product names, status typos, "
            "irrelevant columns (_internal_id, debug_flag, legacy_code), revenue as strings "
            "with dollar signs, duplicates, missing values, and schema inconsistencies "
            "(unit_price renamed to unit_cost in some rows). Goal: Fix all issues while "
            "preserving valid data."
        ),
        "generator": _generate_hard_dataset,
        "max_steps": 35,
        "difficulty": "hard",
    },
}


class DataCleaningEnvironment(Environment):
    """OpenEnv-compliant environment for data cleaning tasks."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data: List[Dict[str, Any]] = []
        self._clean_data: List[Dict[str, Any]] = []
        self._issues: Dict = {}
        self._task_id: str = "task_easy"
        self._task_config: Dict = TASK_CONFIGS["task_easy"]
        self._step_count: int = 0
        self._max_steps: int = 15
        self._done: bool = False
        self._episode_id: str = ""
        self._last_action_msg: str = ""
        self._last_action_success: bool = True
        self._reward: float = 0.0
        self._prev_score: float = 0.0
        self._actions_taken: List[str] = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, 
              task_id: Optional[str] = None, **kwargs) -> DataCleaningObservation:
        """Reset the environment with a specific task."""
        if seed is None:
            seed = random.randint(0, 2**31)
        
        self._task_id = task_id or kwargs.get("task_id", "task_easy")
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "task_easy"
        
        self._task_config = TASK_CONFIGS[self._task_id]
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._max_steps = self._task_config["max_steps"]
        self._done = False
        self._reward = 0.0
        self._prev_score = 0.0
        self._last_action_msg = "Environment reset. Inspect the dataset and begin cleaning."
        self._last_action_success = True
        self._actions_taken = []
        
        # Generate dirty dataset
        generator = self._task_config["generator"]
        self._data, self._clean_data, self._issues = generator(seed)
        
        return self._build_observation()

    def step(self, action: DataCleaningAction, timeout_s: Optional[float] = None, 
             **kwargs) -> DataCleaningObservation:
        """Execute a cleaning action on the dataset."""
        if self._done:
            return self._build_observation()
        
        self._step_count += 1
        self._actions_taken.append(action.action_type)
        
        try:
            self._execute_action(action)
            self._last_action_success = True
        except Exception as e:
            self._last_action_success = False
            self._last_action_msg = f"Action failed: {str(e)}"
        
        # Compute reward as delta in quality score
        current_score = _compute_quality_score(self._data, self._clean_data, self._task_id, self._issues)
        delta = current_score - self._prev_score
        
        # Reward = score improvement + small penalty for each step (encourages efficiency)
        self._reward = delta - 0.005
        
        # Bonus for submitting when score is high
        if action.action_type == "submit":
            self._done = True
            self._reward = current_score  # Final reward is the quality score
            self._last_action_msg = f"Dataset submitted! Final quality score: {current_score:.4f}"
        
        # Check step limit
        if self._step_count >= self._max_steps:
            self._done = True
            self._reward = current_score * 0.8  # Penalize for running out of steps
            self._last_action_msg = f"Step limit reached. Auto-submitted. Score: {current_score:.4f} (penalized to {current_score * 0.8:.4f})"
        
        self._prev_score = current_score
        
        return self._build_observation()

    @property
    def state(self) -> DataCleaningState:
        """Return the current environment state."""
        current_score = _compute_quality_score(self._data, self._clean_data, self._task_id, self._issues)
        return DataCleaningState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            current_step=self._step_count,
            max_steps=self._max_steps,
            is_done=self._done,
            current_score=current_score,
            actions_taken=self._actions_taken,
        )

    def _build_observation(self) -> DataCleaningObservation:
        """Build an observation from the current dataset state."""
        # Column info
        all_cols = set()
        for row in self._data:
            all_cols.update(row.keys())
        col_names = sorted(all_cols)
        
        # Column types
        col_types = {}
        for col in col_names:
            types_seen = set()
            for row in self._data:
                if col in row and row[col] is not None:
                    types_seen.add(type(row[col]).__name__)
            col_types[col] = "/".join(sorted(types_seen)) if types_seen else "unknown"
        
        # Missing values
        missing_counts = {}
        for col in col_names:
            count = sum(1 for row in self._data if col not in row or row.get(col) is None)
            if count > 0:
                missing_counts[col] = count
        
        # Duplicate count
        seen = set()
        dup_count = 0
        for row in self._data:
            key = str(sorted(row.items()))
            if key in seen:
                dup_count += 1
            seen.add(key)
        
        # Sample data
        sample = self._data[:5] if self._data else []
        safe_sample = []
        for row in sample:
            safe_row = {}
            for k, v in row.items():
                if v is None:
                    safe_row[k] = None
                elif isinstance(v, (int, float, bool, str)):
                    safe_row[k] = v
                else:
                    safe_row[k] = str(v)
            safe_sample.append(safe_row)
        
        # Column stats for numeric columns
        col_stats = {}
        for col in col_names:
            vals = []
            for row in self._data:
                if col in row and row.get(col) is not None:
                    try:
                        v = row[col]
                        if isinstance(v, str):
                            v = v.replace("$", "").replace(",", "")
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if len(vals) >= 3:
                mean_val = sum(vals) / len(vals)
                std_val = math.sqrt(sum((x - mean_val) ** 2 for x in vals) / len(vals))
                col_stats[col] = {
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "count": len(vals),
                }
        
        # Detect issues
        detected = []
        if dup_count > 0:
            detected.append(f"Found {dup_count} duplicate rows")
        for col, count in missing_counts.items():
            detected.append(f"Column '{col}' has {count} missing values")
        for col, types_str in col_types.items():
            if "/" in types_str:
                detected.append(f"Column '{col}' has mixed types: {types_str}")
        
        # Check for specific issues per task
        if self._task_id in ("task_medium", "task_hard"):
            date_cols = [c for c in col_names if "date" in c.lower()]
            for dc in date_cols:
                formats_seen = set()
                for row in self._data:
                    if dc in row and row[dc] is not None:
                        val = str(row[dc])
                        if "/" in val:
                            formats_seen.add("slash")
                        elif val.count("-") == 2 and len(val) == 10 and val[4] == "-":
                            formats_seen.add("iso")
                        elif "." in val:
                            formats_seen.add("dot")
                        else:
                            formats_seen.add("other")
                if len(formats_seen) > 1:
                    detected.append(f"Column '{dc}' has inconsistent date formats: {formats_seen}")
        
        if self._task_id == "task_hard":
            irrelevant = {"_internal_id", "debug_flag", "legacy_code"}
            found_irrelevant = all_cols & irrelevant
            if found_irrelevant:
                detected.append(f"Found potentially irrelevant columns: {found_irrelevant}")
            
            # Check for negative quantities
            neg_qty = sum(1 for row in self._data if "quantity" in row and row.get("quantity") is not None 
                         and isinstance(row["quantity"], (int, float)) and row["quantity"] < 0)
            if neg_qty > 0:
                detected.append(f"Found {neg_qty} rows with negative quantities")
        
        current_score = _compute_quality_score(self._data, self._clean_data, self._task_id, self._issues)
        
        return DataCleaningObservation(
            done=self._done,
            reward=self._reward,
            num_rows=len(self._data),
            num_columns=len(col_names),
            column_names=col_names,
            column_types=col_types,
            missing_value_counts=missing_counts,
            duplicate_row_count=dup_count,
            sample_data=safe_sample,
            column_stats=col_stats,
            detected_issues=detected,
            last_action_success=self._last_action_success,
            last_action_message=self._last_action_msg,
            task_id=self._task_id,
            task_description=self._task_config["description"],
            max_steps=self._max_steps,
            current_step=self._step_count,
            metadata={"quality_score": current_score, "episode_id": self._episode_id},
        )

    def _execute_action(self, action: DataCleaningAction):
        """Execute a data cleaning action on the dataset."""
        action_type = action.action_type
        column = action.column
        params = action.params
        
        if action_type == "remove_duplicates":
            self._action_remove_duplicates(column, params)
        elif action_type == "fill_missing":
            self._action_fill_missing(column, params)
        elif action_type == "standardize_format":
            self._action_standardize_format(column, params)
        elif action_type == "fix_outliers":
            self._action_fix_outliers(column, params)
        elif action_type == "rename_column":
            self._action_rename_column(column, params)
        elif action_type == "drop_column":
            self._action_drop_column(column, params)
        elif action_type == "correct_typos":
            self._action_correct_typos(column, params)
        elif action_type == "convert_type":
            self._action_convert_type(column, params)
        elif action_type == "submit":
            pass  # Handled in step()
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _action_remove_duplicates(self, column: Optional[str], params: Dict):
        """Remove duplicate rows."""
        before = len(self._data)
        seen = set()
        unique_rows = []
        for row in self._data:
            if column:
                key = str(row.get(column))
            else:
                key = str(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
        self._data = unique_rows
        removed = before - len(self._data)
        self._last_action_msg = f"Removed {removed} duplicate rows. Rows: {before}  {len(self._data)}"

    def _action_fill_missing(self, column: Optional[str], params: Dict):
        """Fill missing values in a column."""
        if not column:
            raise ValueError("Column name required for fill_missing action")
        
        strategy = params.get("strategy", "drop")  # drop, fill_value, mean, median, mode
        fill_value = params.get("fill_value", None)
        
        filled = 0
        if strategy == "drop":
            before = len(self._data)
            self._data = [row for row in self._data if column in row and row.get(column) is not None]
            filled = before - len(self._data)
            self._last_action_msg = f"Dropped {filled} rows with missing '{column}'"
        elif strategy == "fill_value":
            for row in self._data:
                if column not in row or row.get(column) is None:
                    row[column] = fill_value
                    filled += 1
            self._last_action_msg = f"Filled {filled} missing values in '{column}' with '{fill_value}'"
        elif strategy == "mean":
            vals = [row[column] for row in self._data if column in row and row.get(column) is not None 
                    and isinstance(row[column], (int, float))]
            if vals:
                mean_val = round(sum(vals) / len(vals), 2)
                for row in self._data:
                    if column not in row or row.get(column) is None:
                        row[column] = mean_val
                        filled += 1
            self._last_action_msg = f"Filled {filled} missing values in '{column}' with mean"
        elif strategy == "mode":
            from collections import Counter
            vals = [row[column] for row in self._data if column in row and row.get(column) is not None]
            if vals:
                mode_val = Counter(vals).most_common(1)[0][0]
                for row in self._data:
                    if column not in row or row.get(column) is None:
                        row[column] = mode_val
                        filled += 1
            self._last_action_msg = f"Filled {filled} missing values in '{column}' with mode"
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")

    def _action_standardize_format(self, column: Optional[str], params: Dict):
        """Standardize format of values in a column."""
        if not column:
            raise ValueError("Column name required for standardize_format action")
        
        target_format = params.get("format", "iso_date")  # iso_date, phone_standard, lowercase, uppercase, titlecase
        changed = 0
        
        if target_format == "iso_date":
            # Convert all dates to YYYY-MM-DD
            for row in self._data:
                if column in row and row[column] is not None:
                    val = str(row[column])
                    try:
                        parts = None
                        if "/" in val:
                            p = val.split("/")
                            if len(p) == 3:
                                if len(p[2]) == 4:  # MM/DD/YYYY
                                    parts = (p[2], p[0], p[1])
                                elif len(p[0]) == 4:  # YYYY/MM/DD
                                    parts = (p[0], p[1], p[2])
                        elif "." in val:
                            p = val.split(".")
                            if len(p) == 3 and len(p[0]) == 4:  # YYYY.MM.DD
                                parts = (p[0], p[1], p[2])
                        elif "-" in val:
                            p = val.split("-")
                            if len(p) == 3:
                                if len(p[0]) == 4:  # YYYY-MM-DD (already good)
                                    parts = (p[0], p[1], p[2])
                                elif len(p[2]) == 4:  # DD-MM-YYYY
                                    parts = (p[2], p[1], p[0])
                        
                        if parts:
                            new_val = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                            if new_val != val:
                                row[column] = new_val
                                changed += 1
                    except (ValueError, IndexError):
                        pass
            self._last_action_msg = f"Standardized {changed} dates in '{column}' to ISO format (YYYY-MM-DD)"
        
        elif target_format == "phone_standard":
            import re
            for row in self._data:
                if column in row and row[column] is not None:
                    digits = re.sub(r'\D', '', str(row[column]))
                    if len(digits) == 10:
                        new_val = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                        if new_val != row[column]:
                            row[column] = new_val
                            changed += 1
            self._last_action_msg = f"Standardized {changed} phone numbers in '{column}'"
        
        elif target_format in ("lowercase", "uppercase", "titlecase"):
            for row in self._data:
                if column in row and isinstance(row[column], str):
                    old_val = row[column]
                    if target_format == "lowercase":
                        row[column] = old_val.lower()
                    elif target_format == "uppercase":
                        row[column] = old_val.upper()
                    elif target_format == "titlecase":
                        row[column] = old_val.title()
                    if row[column] != old_val:
                        changed += 1
            self._last_action_msg = f"Standardized {changed} values in '{column}' to {target_format}"
        
        elif target_format == "strip_whitespace":
            import re
            for row in self._data:
                if column in row and isinstance(row[column], str):
                    old_val = row[column]
                    row[column] = re.sub(r'\s+', ' ', row[column]).strip()
                    # Also replace non-breaking spaces
                    row[column] = row[column].replace('\u00a0', ' ')
                    row[column] = re.sub(r'\s+', ' ', row[column]).strip()
                    if row[column] != old_val:
                        changed += 1
            self._last_action_msg = f"Stripped extra whitespace from {changed} values in '{column}'"
        
        else:
            raise ValueError(f"Unknown format: {target_format}")

    def _action_fix_outliers(self, column: Optional[str], params: Dict):
        """Fix outliers in a numeric column."""
        if not column:
            raise ValueError("Column name required for fix_outliers action")
        
        strategy = params.get("strategy", "clip")  # clip, remove, replace_mean
        lower = params.get("lower_bound", None)
        upper = params.get("upper_bound", None)
        
        # Calculate bounds if not provided
        vals = []
        for row in self._data:
            if column in row and row.get(column) is not None:
                try:
                    v = row[column]
                    if isinstance(v, str):
                        v = v.replace("$", "").replace(",", "")
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        
        if vals and (lower is None or upper is None):
            mean_val = sum(vals) / len(vals)
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0
            if lower is None:
                lower = mean_val - 3 * std_val
            if upper is None:
                upper = mean_val + 3 * std_val
        
        fixed = 0
        if strategy == "clip":
            for row in self._data:
                if column in row and row.get(column) is not None:
                    try:
                        v = row[column]
                        if isinstance(v, str):
                            v = float(v.replace("$", "").replace(",", ""))
                        v = float(v)
                        if v < lower:
                            row[column] = round(lower, 2)
                            fixed += 1
                        elif v > upper:
                            row[column] = round(upper, 2)
                            fixed += 1
                    except (ValueError, TypeError):
                        pass
        elif strategy == "remove":
            before = len(self._data)
            new_data = []
            for row in self._data:
                if column in row and row.get(column) is not None:
                    try:
                        v = row[column]
                        if isinstance(v, str):
                            v = float(v.replace("$", "").replace(",", ""))
                        v = float(v)
                        if lower <= v <= upper:
                            new_data.append(row)
                        else:
                            fixed += 1
                    except (ValueError, TypeError):
                        new_data.append(row)
                else:
                    new_data.append(row)
            self._data = new_data
        
        self._last_action_msg = f"Fixed {fixed} outliers in '{column}' using {strategy} strategy (bounds: [{lower:.2f}, {upper:.2f}])"

    def _action_rename_column(self, column: Optional[str], params: Dict):
        """Rename a column."""
        if not column:
            raise ValueError("Column name required for rename_column action")
        new_name = params.get("new_name")
        if not new_name:
            raise ValueError("'new_name' parameter required for rename_column action")
        
        changed = 0
        for row in self._data:
            if column in row:
                row[new_name] = row.pop(column)
                changed += 1
        self._last_action_msg = f"Renamed column '{column}' to '{new_name}' in {changed} rows"

    def _action_drop_column(self, column: Optional[str], params: Dict):
        """Drop a column from the dataset."""
        if not column:
            raise ValueError("Column name required for drop_column action")
        
        dropped = 0
        for row in self._data:
            if column in row:
                del row[column]
                dropped += 1
        self._last_action_msg = f"Dropped column '{column}' from {dropped} rows"

    def _action_correct_typos(self, column: Optional[str], params: Dict):
        """Fix typos/inconsistencies in a categorical column."""
        if not column:
            raise ValueError("Column name required for correct_typos action")
        
        mapping = params.get("mapping", {})
        if not mapping:
            raise ValueError("'mapping' parameter required for correct_typos (dict of old_value -> new_value)")
        
        fixed = 0
        for row in self._data:
            if column in row and row[column] is not None:
                val = row[column]
                # Check exact match
                if val in mapping:
                    row[column] = mapping[val]
                    fixed += 1
                # Check case-insensitive match
                elif isinstance(val, str):
                    for old, new in mapping.items():
                        if isinstance(old, str) and val.lower().strip() == old.lower().strip():
                            row[column] = new
                            fixed += 1
                            break
        self._last_action_msg = f"Fixed {fixed} typos in '{column}' using provided mapping"

    def _action_convert_type(self, column: Optional[str], params: Dict):
        """Convert column values to a specific type."""
        if not column:
            raise ValueError("Column name required for convert_type action")
        
        target_type = params.get("target_type", "float")  # int, float, str, bool
        strip_chars = params.get("strip_chars", "")  # e.g., "$," for currency
        
        converted = 0
        errors = 0
        for row in self._data:
            if column in row and row.get(column) is not None:
                val = row[column]
                try:
                    if isinstance(val, str) and strip_chars:
                        for ch in strip_chars:
                            val = val.replace(ch, "")
                    
                    if target_type == "float":
                        row[column] = round(float(val), 2)
                    elif target_type == "int":
                        row[column] = int(float(val))
                    elif target_type == "str":
                        row[column] = str(val)
                    elif target_type == "bool":
                        row[column] = bool(val)
                    converted += 1
                except (ValueError, TypeError):
                    errors += 1
        
        self._last_action_msg = f"Converted {converted} values in '{column}' to {target_type} ({errors} errors)"
