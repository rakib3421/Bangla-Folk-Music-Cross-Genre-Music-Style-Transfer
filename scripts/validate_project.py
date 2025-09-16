#!/usr/bin/env python3
"""
Project Organization Validation Script

This script validates that the Bangla Folk to Rock/Jazz Style Transfer System
is properly organized and ready for GitHub publication.
"""

import os
import sys
from pathlib import Path
import subprocess
import importlib.util

class ProjectValidator:
    """Validates project organization and setup."""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.issues = []
        self.successes = []
    
    def validate_directory_structure(self):
        """Validate that all required directories exist."""
        print("🔍 Validating directory structure...")
        
        required_dirs = [
            "src",
            "src/audio", 
            "src/models",
            "src/training",
            "src/evaluation", 
            "src/interactive",
            "tests",
            "scripts",
            "docs",
            ".github",
            ".github/workflows"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.successes.append(f"✅ Directory exists: {dir_path}")
            else:
                self.issues.append(f"❌ Missing directory: {dir_path}")
    
    def validate_required_files(self):
        """Validate that all required files exist."""
        print("📋 Validating required files...")
        
        required_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            ".gitignore",
            "dev_config.py",
            "src/__init__.py",
            "src/audio/__init__.py",
            "src/models/__init__.py", 
            "src/training/__init__.py",
            "src/evaluation/__init__.py",
            "src/interactive/__init__.py",
            "tests/__init__.py",
            "docs/API.md",
            "docs/CONTRIBUTING.md",
            ".github/workflows/ci.yml"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.successes.append(f"✅ File exists: {file_path}")
            else:
                self.issues.append(f"❌ Missing file: {file_path}")
    
    def validate_python_modules(self):
        """Validate that Python modules are properly organized."""
        print("🐍 Validating Python modules...")
        
        module_dirs = [
            "src/audio",
            "src/models", 
            "src/training",
            "src/evaluation",
            "src/interactive"
        ]
        
        for module_dir in module_dirs:
            full_path = self.project_root / module_dir
            if full_path.exists():
                py_files = list(full_path.glob("*.py"))
                py_files = [f for f in py_files if f.name != "__init__.py"]
                
                if py_files:
                    self.successes.append(f"✅ {module_dir} contains {len(py_files)} Python modules")
                else:
                    self.issues.append(f"⚠️  {module_dir} contains no Python modules")
    
    def validate_imports(self):
        """Validate that package imports work correctly."""
        print("📦 Validating package imports...")
        
        # Add src to Python path for testing
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        test_imports = [
            "audio",
            "models", 
            "training",
            "evaluation",
            "interactive"
        ]
        
        for module_name in test_imports:
            try:
                importlib.import_module(module_name)
                self.successes.append(f"✅ Successfully imported: {module_name}")
            except ImportError as e:
                self.issues.append(f"❌ Failed to import {module_name}: {e}")
    
    def validate_git_setup(self):
        """Validate Git repository setup."""
        print("🔧 Validating Git setup...")
        
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            self.successes.append("✅ Git repository initialized")
            
            # Check for staged/unstaged changes
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.stdout.strip():
                    self.issues.append("⚠️  Uncommitted changes detected")
                else:
                    self.successes.append("✅ Working directory clean")
                    
            except subprocess.CalledProcessError:
                self.issues.append("❌ Failed to check git status")
        else:
            self.issues.append("❌ Git repository not initialized")
    
    def validate_requirements(self):
        """Validate requirements.txt content."""
        print("📋 Validating requirements...")
        
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read().strip()
                
                essential_packages = [
                    "torch", "torchaudio", "librosa", "numpy", 
                    "scipy", "soundfile", "matplotlib"
                ]
                
                missing_packages = []
                for package in essential_packages:
                    if package not in requirements:
                        missing_packages.append(package)
                
                if missing_packages:
                    self.issues.append(f"⚠️  Missing essential packages: {', '.join(missing_packages)}")
                else:
                    self.successes.append("✅ All essential packages listed in requirements")
                    
            except Exception as e:
                self.issues.append(f"❌ Failed to read requirements.txt: {e}")
        else:
            self.issues.append("❌ requirements.txt not found")
    
    def generate_report(self):
        """Generate a comprehensive validation report."""
        print("\n" + "="*60)
        print("📊 PROJECT VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ SUCCESSES ({len(self.successes)}):")
        for success in self.successes:
            print(f"  {success}")
        
        print(f"\n❌ ISSUES ({len(self.issues)}):")
        for issue in self.issues:
            print(f"  {issue}")
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total checks: {len(self.successes) + len(self.issues)}")
        print(f"  Successful: {len(self.successes)}")
        print(f"  Issues found: {len(self.issues)}")
        
        if len(self.issues) == 0:
            print("\n🎉 PROJECT IS READY FOR GITHUB PUBLICATION!")
            print("✨ All validation checks passed successfully.")
        elif len(self.issues) <= 3:
            print("\n⚠️  PROJECT IS MOSTLY READY")
            print("🔧 Please address the minor issues above.")
        else:
            print("\n🚫 PROJECT NEEDS MORE WORK")
            print("🛠️  Please address the issues before publication.")
        
        return len(self.issues)
    
    def run_all_validations(self):
        """Run all validation checks."""
        print("🚀 Starting project validation...")
        print(f"📂 Project root: {self.project_root}")
        
        self.validate_directory_structure()
        self.validate_required_files()
        self.validate_python_modules()
        self.validate_imports()
        self.validate_git_setup() 
        self.validate_requirements()
        
        return self.generate_report()


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate project organization for GitHub publication"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        help="Path to project root directory",
        default=None
    )
    parser.add_argument(
        "--fix-imports",
        action="store_true",
        help="Attempt to fix import issues automatically"
    )
    
    args = parser.parse_args()
    
    validator = ProjectValidator(args.project_root)
    num_issues = validator.run_all_validations()
    
    # Exit with error code if issues found
    sys.exit(min(num_issues, 1))


if __name__ == "__main__":
    main()