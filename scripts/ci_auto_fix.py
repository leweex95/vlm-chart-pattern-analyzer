#!/usr/bin/env python3
"""
Auto-detect console errors and suggest fixes.
"""
import re
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ErrorPattern:
    """Error pattern definition."""
    pattern: str
    error_type: str
    description: str
    fix_suggestion: str


# Known error patterns and their fixes
ERROR_PATTERNS = [
    ErrorPattern(
        pattern=r"FileNotFoundError.*No such file or directory.*'([^']+)'",
        error_type="FileNotFoundError",
        description="File not found",
        fix_suggestion="Create the missing file or check the file path",
    ),
    ErrorPattern(
        pattern=r"ModuleNotFoundError.*No module named '([^']+)'",
        error_type="ModuleNotFoundError",
        description="Missing Python module",
        fix_suggestion="Install the module with: poetry add {module}",
    ),
    ErrorPattern(
        pattern=r"ImportError.*cannot import name '([^']+)'",
        error_type="ImportError",
        description="Import error",
        fix_suggestion="Check if the import exists in the module",
    ),
    ErrorPattern(
        pattern=r"KeyError.*'([^']+)'",
        error_type="KeyError",
        description="Missing dictionary key",
        fix_suggestion="Check if the key exists or add default value",
    ),
    ErrorPattern(
        pattern=r"AttributeError.*'([^']+)' object has no attribute '([^']+)'",
        error_type="AttributeError",
        description="Missing attribute",
        fix_suggestion="Check if the attribute name is correct",
    ),
    ErrorPattern(
        pattern=r"TypeError.*got an unexpected keyword argument '([^']+)'",
        error_type="TypeError",
        description="Invalid keyword argument",
        fix_suggestion="Remove or correct the keyword argument",
    ),
    ErrorPattern(
        pattern=r"ValueError.*invalid literal for ([^:]+): '([^']+)'",
        error_type="ValueError",
        description="Invalid value conversion",
        fix_suggestion="Check the value format before conversion",
    ),
    ErrorPattern(
        pattern=r"KeyboardInterrupt",
        error_type="KeyboardInterrupt",
        description="User interrupted execution",
        fix_suggestion="Script was interrupted by user (Ctrl+C)",
    ),
    ErrorPattern(
        pattern=r"CUDA out of memory",
        error_type="OutOfMemoryError",
        description="GPU out of memory",
        fix_suggestion="Reduce batch size or use quantization",
    ),
    ErrorPattern(
        pattern=r"RuntimeError.*expected.*got.*tensor",
        error_type="RuntimeError",
        description="Tensor shape mismatch",
        fix_suggestion="Check tensor dimensions and device placement",
    ),
    ErrorPattern(
        pattern=r"Unexpected UTF-8 BOM.*decode using utf-8-sig",
        error_type="JSONDecodeError",
        description="UTF-8 BOM in JSON file",
        fix_suggestion="Remove BOM from file using UTF-8 without BOM encoding",
    ),
    ErrorPattern(
        pattern=r"json\.decoder\.JSONDecodeError",
        error_type="JSONDecodeError",
        description="JSON parsing error",
        fix_suggestion="Check JSON file syntax and encoding",
    ),
]


def detect_errors(console_output: str) -> List[dict]:
    """Detect errors in console output."""
    detected_errors = []
    
    for pattern_obj in ERROR_PATTERNS:
        matches = re.finditer(pattern_obj.pattern, console_output, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            error_info = {
                "type": pattern_obj.error_type,
                "description": pattern_obj.description,
                "fix_suggestion": pattern_obj.fix_suggestion,
                "match": match.group(0),
                "groups": match.groups() if match.groups() else None,
            }
            
            # Format fix suggestion with captured groups
            if error_info["groups"]:
                if pattern_obj.error_type == "ModuleNotFoundError":
                    error_info["fix_suggestion"] = pattern_obj.fix_suggestion.format(
                        module=error_info["groups"][0]
                    )
            
            detected_errors.append(error_info)
    
    return detected_errors


def analyze_console_output(console_output: str, verbose: bool = False) -> None:
    """Analyze console output for errors and suggest fixes."""
    print("=" * 70)
    print("CONSOLE ERROR ANALYSIS")
    print("=" * 70)
    
    errors = detect_errors(console_output)
    
    if not errors:
        print("✓ No errors detected in console output")
        return
    
    print(f"\n⚠ Found {len(errors)} error(s):\n")
    
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error['type']}: {error['description']}")
        print(f"   Match: {error['match'][:100]}...")
        if error['groups']:
            print(f"   Details: {error['groups']}")
        print(f"   Fix: {error['fix_suggestion']}")
        print()
    
    if verbose:
        print("\nFull console output:")
        print("-" * 70)
        print(console_output)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-detect console errors")
    parser.add_argument("--input", "-i", help="Input file with console output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.input:
        with open(args.input, "r") as f:
            console_output = f.read()
    else:
        # Read from stdin
        console_output = sys.stdin.read()
    
    analyze_console_output(console_output, verbose=args.verbose)


if __name__ == "__main__":
    main()
