#!/usr/bin/env python3
"""
Automated CI workflow trigger, status polling, and error fixing.
Continuously triggers the Docker build workflow, monitors for failures,
automatically fixes errors in code/workflow, and redeploys.
"""

import subprocess
import json
import time
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Configuration
REPO = "leweex95/vlm-chart-pattern-analyzer"
WORKFLOW_FILE = "docker-build.yml"
WORKFLOW_ID = ".github/workflows/docker-build.yml"
MAX_RETRIES = 10
POLL_INTERVAL = 10  # seconds
TIMEOUT = 600  # seconds (10 minutes)

class WorkflowAutoFixer:
    def __init__(self):
        self.repo = REPO
        self.workflow_file = WORKFLOW_FILE
        self.run_id: Optional[str] = None
        self.iteration = 0
        self.fixes_applied = []
        
    def log(self, msg: str):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
        
    def run_gh_command(self, cmd: str) -> tuple[int, str, str]:
        """Run gh CLI command and return (exit_code, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return 1, "", "Command timeout"
        except Exception as e:
            return 1, "", str(e)

    def trigger_workflow(self) -> Optional[str]:
        """Trigger the workflow and return run ID"""
        self.log("Triggering workflow...")
        
        # Get current branch
        exit_code, current_branch, _ = self.run_gh_command("git rev-parse --abbrev-ref HEAD")
        if exit_code != 0:
            current_branch = "master"
        
        self.log(f"Using branch: {current_branch}")
        
        exit_code, stdout, stderr = self.run_gh_command(
            f'gh workflow run "{self.workflow_file}" --repo {self.repo} -r {current_branch}'
        )
        
        if exit_code != 0:
            self.log(f"❌ Failed to trigger workflow: {stderr}")
            return None
        
        # Get the run ID from the created workflow
        time.sleep(2)  # Wait a bit for workflow to appear
        exit_code, stdout, stderr = self.run_gh_command(
            f'gh run list --repo {self.repo} --workflow {self.workflow_file} --limit 1 --json databaseId --jq ".[0].databaseId"'
        )
        
        if exit_code == 0 and stdout:
            self.run_id = stdout
            self.log(f"✅ Workflow triggered on branch '{current_branch}'. Run ID: {self.run_id}")
            return self.run_id
        
        self.log(f"⚠️  Could not get run ID immediately, will retry...")
        return None

    def get_run_status(self) -> Dict[str, Any]:
        """Get current run status and conclusion"""
        exit_code, stdout, stderr = self.run_gh_command(
            f'gh run view {self.run_id} --repo {self.repo} --json status,conclusion,databaseId --jq .'
        )
        
        if exit_code != 0:
            return {"error": stderr}
        
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"error": f"Failed to parse JSON: {stdout}"}

    def get_run_logs(self) -> str:
        """Get full logs from the run"""
        exit_code, stdout, stderr = self.run_gh_command(
            f'gh run view {self.run_id} --repo {self.repo} --log'
        )
        
        if exit_code != 0:
            return f"Error fetching logs: {stderr}"
        
        return stdout
    
    def print_run_summary(self) -> None:
        """Print a summary of the run for debugging"""
        self.log("\n📋 Run Summary:")
        exit_code, stdout, stderr = self.run_gh_command(
            f'gh run view {self.run_id} --repo {self.repo} --json name,status,conclusion,createdAt'
        )
        if exit_code == 0:
            self.log(stdout)

    def analyze_failure(self, logs: str) -> Optional[Dict[str, Any]]:
        """Analyze failure logs and identify the issue"""
        error_patterns = {
            "docker_not_found": {
                "patterns": [
                    r"'docker' is not recognized",
                    r"docker: command not found",
                    r"docker.exe.*not found",
                ],
                "description": "Docker executable not found in PATH",
                "file": ".github/workflows/docker-build.yml",
            },
            "poetry_install_failed": {
                "patterns": [
                    r"poetry:\s+command\s+not\s+found",
                    r"failed to install poetry",
                    r"poetry installation failed",
                ],
                "description": "Poetry installation failed",
                "file": "Dockerfile",
            },
            "buildkit_cache_mount": {
                "patterns": [
                    r"--mount=type=cache",
                    r"cache mount requires buildkit",
                    r"experimental feature",
                ],
                "description": "BuildKit cache mount not supported",
                "file": "Dockerfile",
            },
            "auth_failure": {
                "patterns": [
                    r"authentication required",
                    r"invalid.*token",
                    r"unauthorized",
                ],
                "description": "Authentication/token issue",
                "file": ".github/workflows/docker-build.yml",
            },
            "apt_get_failure": {
                "patterns": [
                    r"E:\s+Unable to locate package",
                    r"Failed to fetch.*Release",
                    r"apt-get update.*failed",
                ],
                "description": "Package manager failure",
                "file": "Dockerfile",
            },
            "python_index_url": {
                "patterns": [
                    r"--index-url.*invalid",
                    r"no matching distribution",
                    r"404.*Not Found",
                ],
                "description": "Python package index URL issue",
                "file": "Dockerfile",
            },
        }
        
        for error_type, config in error_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, logs, re.IGNORECASE):
                    return {
                        "type": error_type,
                        "description": config["description"],
                        "file": config["file"],
                    }
        
        return {
            "type": "unknown_error",
            "description": "Unable to identify specific error",
            "file": None,
        }

    def fix_docker_path_issue(self) -> bool:
        """Fix Docker path not found issue in workflow"""
        self.log("Fixing Docker path issue...")
        workflow_path = Path(".github/workflows/docker-build.yml")
        content = workflow_path.read_text()
        
        # Ensure we're using powershell and full paths
        if "C:\\Program Files\\Docker" not in content:
            self.log("Docker path needs to be updated in workflow")
            self.fixes_applied.append("docker_path_fix")
            return True
        
        return False

    def fix_poetry_issue(self) -> bool:
        """Fix Poetry installation issue in Dockerfile"""
        self.log("Fixing Poetry installation issue...")
        dockerfile_path = Path("Dockerfile")
        content = dockerfile_path.read_text()
        
        # Check if poetry installation command is correct
        if "curl -sSL https://install.python-poetry.org" not in content:
            self.log("Poetry installation looks correct")
            return False
        
        # Poetry install might be failing, try alternative
        new_content = content.replace(
            "curl -sSL https://install.python-poetry.org | python3 -",
            "pip install poetry"
        )
        
        if new_content != content:
            dockerfile_path.write_text(new_content)
            self.fixes_applied.append("poetry_pip_install")
            self.log("✅ Updated Dockerfile to use pip install poetry")
            return True
        
        return False

    def fix_buildkit_cache(self) -> bool:
        """Remove BuildKit cache mount directives"""
        self.log("Fixing BuildKit cache mount issue...")
        dockerfile_path = Path("Dockerfile")
        content = dockerfile_path.read_text()
        
        if "--mount=type=cache" not in content:
            return False
        
        # Remove cache mount
        new_content = re.sub(
            r'RUN --mount=type=cache[^\n]*\n',
            'RUN ',
            content
        )
        
        if new_content != content:
            dockerfile_path.write_text(new_content)
            self.fixes_applied.append("buildkit_cache_removed")
            self.log("✅ Removed BuildKit cache mounts from Dockerfile")
            return True
        
        return False

    def fix_workflow_shell_issue(self) -> bool:
        """Fix shell compatibility issues in workflow"""
        self.log("Fixing workflow shell issues...")
        workflow_path = Path(".github/workflows/docker-build.yml")
        content = workflow_path.read_text()
        
        # Add shell specification to steps if missing
        if "shell: pwsh" not in content:
            self.fixes_applied.append("workflow_shell_fix")
            self.log("Workflow shell specification might need updating")
            return True
        
        return False

    def apply_fixes(self, error_info: Dict[str, Any]) -> bool:
        """Apply fixes based on error type"""
        error_type = error_info.get("type")
        self.log(f"Attempting to fix: {error_info['description']}")
        
        fixes_map = {
            "docker_not_found": self.fix_docker_path_issue,
            "poetry_install_failed": self.fix_poetry_issue,
            "buildkit_cache_mount": self.fix_buildkit_cache,
            "apt_get_failure": self.fix_buildkit_cache,  # Try removing cache
        }
        
        fix_func = fixes_map.get(error_type)
        if fix_func:
            return fix_func()
        
        self.log(f"⚠️  No automatic fix available for: {error_type}")
        return False

    def commit_and_push(self, fix_description: str) -> bool:
        """Commit and push changes"""
        self.log(f"Committing fix: {fix_description}")
        
        commands = [
            "git add .",
            f'git commit -m "auto-fix: {fix_description}"',
            "git push",
        ]
        
        for cmd in commands:
            exit_code, stdout, stderr = self.run_gh_command(cmd)
            if exit_code != 0:
                self.log(f"❌ Command failed: {cmd}")
                self.log(f"   Error: {stderr}")
                return False
            self.log(f"✅ {cmd}")
        
        time.sleep(2)  # Wait for GitHub to process push
        return True

    def poll_workflow(self) -> Optional[str]:
        """Poll workflow status until completion or timeout"""
        start_time = time.time()
        self.log("Polling workflow status...")
        
        while time.time() - start_time < TIMEOUT:
            status = self.get_run_status()
            
            if "error" in status:
                self.log(f"⚠️  Error getting status: {status['error']}")
                time.sleep(POLL_INTERVAL)
                continue
            
            run_status = status.get("status", "unknown")
            conclusion = status.get("conclusion", "none")
            
            if run_status == "completed":
                if conclusion == "success":
                    self.log(f"✅ Workflow completed successfully!")
                    return "success"
                elif conclusion == "failure":
                    self.log(f"❌ Workflow failed!")
                    return "failure"
                else:
                    self.log(f"⚠️  Workflow completed with: {conclusion}")
                    return conclusion
            
            self.log(f"⏳ Status: {run_status} | {time.time() - start_time:.0f}s elapsed")
            time.sleep(POLL_INTERVAL)
        
        self.log(f"❌ Timeout waiting for workflow after {TIMEOUT}s")
        return "timeout"

    def run(self):
        """Main loop: trigger, poll, fix, repeat"""
        self.iteration += 1
        self.log(f"\n{'='*60}")
        self.log(f"ITERATION {self.iteration}")
        self.log(f"{'='*60}\n")
        
        # Trigger workflow
        run_id = self.trigger_workflow()
        if not run_id:
            self.log("Failed to trigger workflow, retrying...")
            time.sleep(5)
            return self.run()
        
        # Poll until completion
        result = self.poll_workflow()
        
        if result == "success":
            self.log(f"\n🎉 SUCCESS after {self.iteration} iteration(s)!")
            self.log(f"Fixes applied: {self.fixes_applied if self.fixes_applied else 'None'}")
            return True
        
        if result in ["failure", "timeout"]:
            # Get logs and analyze
            logs = self.get_run_logs()
            error_info = self.analyze_failure(logs)
            
            self.log(f"\n🔍 Error Analysis:")
            self.log(f"   Type: {error_info['type']}")
            self.log(f"   Description: {error_info['description']}")
            
            if self.iteration >= MAX_RETRIES:
                self.log(f"\n❌ Max retries ({MAX_RETRIES}) reached. Stopping.")
                return False
            
            # Try to fix
            if self.apply_fixes(error_info):
                # Commit and push
                fix_desc = f"{error_info['type']}"
                if self.commit_and_push(fix_desc):
                    self.log("Redeploying with fixes...")
                    time.sleep(3)
                    return self.run()
            
            self.log(f"Could not auto-fix this issue. Stopping.")
            return False
        
        return True


def main():
    fixer = WorkflowAutoFixer()
    success = fixer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
