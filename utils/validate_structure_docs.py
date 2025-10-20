#!/usr/bin/env python3
"""
Validate Structure Documentation Consistency

This script ensures that all Python scripts in src/ have:
1. The canonical experiment folder structure documented in their docstrings
2. Validation calls to validate_experiment_structure()

Usage:
    python utils/validate_structure_docs.py
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


# Expected structure marker in docstrings
STRUCTURE_MARKER = "Expected Experiment Folder Structure:"
VALIDATION_IMPORT = "from src.experiment_structure import validate_experiment_structure"
VALIDATION_CALL = "validate_experiment_structure("


def check_file(file_path: Path) -> Dict[str, bool]:
    """
    Check if a file has proper structure documentation and validation.

    Args:
        file_path: Path to Python file

    Returns:
        Dictionary with check results
    """
    content = file_path.read_text()

    # Extract module docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    docstring = docstring_match.group(1) if docstring_match else ""

    results = {
        "has_structure_docs": STRUCTURE_MARKER in docstring,
        "has_validation_import": VALIDATION_IMPORT in content,
        "has_validation_call": VALIDATION_CALL in content,
        "is_main_script": "def main(" in content or 'if __name__ == "__main__"' in content
    }

    return results


def main():
    """Validate structure documentation across all src/ scripts."""
    src_dir = Path(__file__).parent.parent / "src"

    # Get all Python files except __init__.py and experiment_structure.py
    python_files = [
        f for f in src_dir.glob("*.py")
        if f.name not in ["__init__.py", "experiment_structure.py"]
    ]

    print("=" * 80)
    print("Validating Experiment Structure Documentation")
    print("=" * 80)
    print()

    all_passed = True
    results_summary = []

    for py_file in sorted(python_files):
        results = check_file(py_file)

        status_parts = []
        issues = []

        # Check structure docs
        if results["has_structure_docs"]:
            status_parts.append("✓ Docs")
        else:
            status_parts.append("✗ Docs")
            issues.append("Missing structure documentation in docstring")

        # Check validation (only for main scripts)
        if results["is_main_script"]:
            if results["has_validation_import"]:
                status_parts.append("✓ Import")
            else:
                status_parts.append("✗ Import")
                issues.append("Missing validation import")

            if results["has_validation_call"]:
                status_parts.append("✓ Call")
            else:
                status_parts.append("✗ Call")
                issues.append("Missing validation call")
        else:
            status_parts.append("- N/A (not main)")

        passed = (results["has_structure_docs"] and
                 (not results["is_main_script"] or
                  (results["has_validation_import"] and results["has_validation_call"])))

        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        results_summary.append({
            "file": py_file.name,
            "status": status,
            "details": " | ".join(status_parts),
            "issues": issues
        })

    # Print results
    for result in results_summary:
        print(f"{result['status']:8s} {result['file']:40s} {result['details']}")
        if result['issues']:
            for issue in result['issues']:
                print(f"         └─ {issue}")
            print()

    print()
    print("=" * 80)

    if all_passed:
        print("✓ All scripts have consistent structure documentation!")
        print()
        print("Next steps:")
        print("1. Run: python -m src.experiment_structure validate <experiment_dir>")
        print("2. Or use validate_experiment_structure() in your scripts")
        return 0
    else:
        print("✗ Some scripts are missing structure documentation or validation!")
        print()
        print("To fix:")
        print("1. Add structure documentation to module docstring:")
        print("   See src/predict.py for an example")
        print("2. Import and call validate_experiment_structure() in main():")
        print(f"   {VALIDATION_IMPORT}")
        print("   validate_experiment_structure(experiment_dir, required_dirs=[...], verbose=True)")
        print()
        print("For reference structure, see: src/experiment_structure.py")
        return 1


if __name__ == "__main__":
    exit(main())
