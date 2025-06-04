#!/usr/bin/env python3
"""
Script to clean up YOLO model weights and push to git repository.
Keeps only best.pt files and removes all other weight files to save space.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

class WeightCleaner:
    def __init__(self):
        self.project_root = Path.cwd()
        self.runs_folders = [
            "runs/runs/train",
            "runs-yolo-v7/runs/train"
        ]
        
    def find_weight_files(self, directory):
        """Find all weight files in a directory."""
        weight_files = []
        weights_dir = Path(directory) / "weights"
        if weights_dir.exists():
            for file in weights_dir.iterdir():
                if file.suffix == '.pt':
                    weight_files.append(file)
        return weight_files
    
    def cleanup_weights_in_folder(self, runs_folder):
        """Clean up weights in a specific runs folder."""
        runs_path = self.project_root / runs_folder
        if not runs_path.exists():
            print(f"❌ Runs folder not found: {runs_folder}")
            return False, 0, 0
        
        print(f"\n🔍 Processing: {runs_folder}")
        total_deleted = 0
        folders_processed = 0
        
        # Find all training experiment folders
        for experiment_folder in runs_path.iterdir():
            if experiment_folder.is_dir():
                print(f"  📁 Checking: {experiment_folder.name}")
                weight_files = self.find_weight_files(experiment_folder)
                
                if weight_files:
                    folders_processed += 1
                    best_file = None
                    files_to_delete = []
                    
                    # Identify best.pt and files to delete
                    for weight_file in weight_files:
                        if weight_file.name == 'best.pt':
                            best_file = weight_file
                        else:
                            files_to_delete.append(weight_file)
                    
                    if best_file:
                        print(f"    ✅ Keeping: {best_file.name} ({self.get_file_size(best_file)})")
                    
                    # Delete non-best weight files
                    for file_to_delete in files_to_delete:
                        try:
                            file_size = self.get_file_size(file_to_delete)
                            file_to_delete.unlink()
                            print(f"    🗑️  Deleted: {file_to_delete.name} ({file_size})")
                            total_deleted += 1
                        except Exception as e:
                            print(f"    ❌ Failed to delete {file_to_delete.name}: {e}")
                
        print(f"  📊 Summary: {folders_processed} folders processed, {total_deleted} files deleted")
        return True, folders_processed, total_deleted
    
    def get_file_size(self, file_path):
        """Get human readable file size."""
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    
    def cleanup_all_weights(self):
        """Clean up weights in all runs folders."""
        print("🧹 Starting weight cleanup process...")
        total_folders = 0
        total_deleted = 0
        
        for runs_folder in self.runs_folders:
            success, folders, deleted = self.cleanup_weights_in_folder(runs_folder)
            if success:
                total_folders += folders
                total_deleted += deleted
        
        print(f"\n✅ Cleanup complete!")
        print(f"📊 Total: {total_folders} model folders processed, {total_deleted} weight files deleted")
        return total_deleted > 0

class GitManager:
    def __init__(self):
        self.project_root = Path.cwd()
    
    def run_git_command(self, command, description=""):
        """Run a git command and return the result."""
        try:
            print(f"🔧 {description}: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            if result.stderr.strip() and result.returncode != 0:
                print(f"   Error: {result.stderr.strip()}")
            
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            print(f"❌ Failed to run command '{command}': {e}")
            return False, "", str(e)
    
    def setup_git_remote(self):
        """Setup git remote if it doesn't exist."""
        success, stdout, stderr = self.run_git_command("git remote -v", "Checking git remotes")
        
        if not stdout.strip():  # No remotes configured
            print("📡 No git remote found. Please add your remote repository:")
            remote_url = input("Enter your git remote URL: ").strip()
            if remote_url:
                return self.run_git_command(f'git remote add origin "{remote_url}"', "Adding git remote")
            else:
                print("❌ No remote URL provided. Skipping git operations.")
                return False, "", ""
        return True, stdout, stderr
    
    def get_current_branch(self):
        """Get the current git branch."""
        success, stdout, stderr = self.run_git_command("git branch --show-current", "Getting current branch")
        if success:
            return stdout.strip()
        return None
    
    def create_new_branch(self, base_name="model-weights"):
        """Create a new branch with incremental naming."""
        current_branch = self.get_current_branch()
        
        # List existing branches
        success, stdout, stderr = self.run_git_command("git branch -a", "Listing branches")
        existing_branches = []
        if success:
            for line in stdout.split('\n'):
                branch = line.strip().replace('* ', '').replace('remotes/origin/', '')
                if branch and not branch.startswith('HEAD'):
                    existing_branches.append(branch)
        
        # Find a unique branch name
        counter = 1
        new_branch = f"{base_name}-{counter}"
        while new_branch in existing_branches:
            counter += 1
            new_branch = f"{base_name}-{counter}"
        
        # Create and checkout new branch
        success, stdout, stderr = self.run_git_command(f"git checkout -b {new_branch}", f"Creating new branch '{new_branch}'")
        return success, new_branch
    
    def update_gitignore(self):
        """Remove weight-related entries from .gitignore."""
        gitignore_path = self.project_root / ".gitignore"
        
        if not gitignore_path.exists():
            print("📝 No .gitignore file found")
            return True
        
        # Read current .gitignore
        with open(gitignore_path, 'r') as f:
            lines = f.readlines()
        
        # Remove lines that ignore weight files but keep runs* pattern
        original_lines = len(lines)
        filtered_lines = []
        removed_patterns = []
        
        for line in lines:
            line_stripped = line.strip()
            # Keep runs* pattern but allow specific weight files
            if line_stripped in ['*.pt', '*.pth', 'weights/', 'weights/*', '**/weights/*']:
                removed_patterns.append(line_stripped)
            else:
                filtered_lines.append(line)
        
        if removed_patterns:
            # Write updated .gitignore
            with open(gitignore_path, 'w') as f:
                f.writelines(filtered_lines)
            
            print(f"📝 Updated .gitignore:")
            for pattern in removed_patterns:
                print(f"   ❌ Removed: {pattern}")
            
            return True
        else:
            print("📝 .gitignore doesn't contain weight-related patterns")
            return False
    
    def commit_and_push_changes(self, runs_folder_name):
        """Add, commit and push changes for a specific runs folder."""
        # Add specific runs folder
        folder_path = runs_folder_name.replace('/', '\\') if os.name == 'nt' else runs_folder_name
        
        success, stdout, stderr = self.run_git_command(f'git add "{folder_path}"', f"Adding {runs_folder_name}")
        if not success:
            print(f"❌ Failed to add {runs_folder_name}")
            return False
        
        # Commit changes
        commit_message = f"Add cleaned model weights from {runs_folder_name}"
        success, stdout, stderr = self.run_git_command(f'git commit -m "{commit_message}"', f"Committing {runs_folder_name}")
        if not success and "nothing to commit" not in stderr:
            print(f"❌ Failed to commit {runs_folder_name}")
            return False
        
        # Push changes
        current_branch = self.get_current_branch()
        if current_branch:
            success, stdout, stderr = self.run_git_command(f"git push origin {current_branch}", f"Pushing {runs_folder_name}")
            if not success:
                print(f"❌ Failed to push {runs_folder_name}")
                # Try setting upstream
                success, stdout, stderr = self.run_git_command(f"git push -u origin {current_branch}", f"Pushing {runs_folder_name} with upstream")
                if not success:
                    print(f"❌ Failed to push {runs_folder_name} even with upstream")
                    return False
        
        print(f"✅ Successfully pushed {runs_folder_name}")
        return True

def main():
    print("🚀 YOLO Model Weight Cleanup and Git Push Tool")
    print("=" * 50)
    
    # Initialize components
    cleaner = WeightCleaner()
    git_manager = GitManager()
    
    # Step 1: Clean up weights
    print("\n📋 Step 1: Cleaning up model weights...")
    weights_deleted = cleaner.cleanup_all_weights()
    
    if not weights_deleted:
        print("ℹ️  No weights were deleted. Checking if we should still proceed with git operations...")
        proceed = input("Do you want to proceed with git operations anyway? (y/N): ").lower().startswith('y')
        if not proceed:
            print("👋 Exiting.")
            return
    
    # Step 2: Setup git remote if needed
    print("\n📋 Step 2: Setting up git configuration...")
    success, stdout, stderr = git_manager.setup_git_remote()
    if not success:
        print("❌ Git remote setup failed. Exiting.")
        return
    
    # Step 3: Create new branch or use existing
    print("\n📋 Step 3: Managing git branches...")
    current_branch = git_manager.get_current_branch()
    
    if current_branch:
        print(f"📍 Current branch: {current_branch}")
        use_current = input("Do you want to use the current branch? (Y/n): ").lower()
        if use_current != 'n':
            branch_to_use = current_branch
        else:
            success, branch_to_use = git_manager.create_new_branch()
            if not success:
                print("❌ Failed to create new branch. Exiting.")
                return
    else:
        success, branch_to_use = git_manager.create_new_branch()
        if not success:
            print("❌ Failed to create new branch. Exiting.")
            return
    
    print(f"🌿 Using branch: {branch_to_use}")
    
    # Step 4: Update .gitignore
    print("\n📋 Step 4: Updating .gitignore...")
    git_manager.update_gitignore()
    
    # Step 5: Process each runs folder separately
    print("\n📋 Step 5: Processing runs folders individually...")
    
    for runs_folder in cleaner.runs_folders:
        if Path(runs_folder).exists():
            print(f"\n🔄 Processing: {runs_folder}")
            success = git_manager.commit_and_push_changes(runs_folder)
            if success:
                print(f"✅ Successfully processed {runs_folder}")
            else:
                print(f"❌ Failed to process {runs_folder}")
                continue_anyway = input(f"Continue with next folder? (Y/n): ").lower()
                if continue_anyway == 'n':
                    break
    
    print("\n🎉 All done!")
    print(f"🌿 All changes have been pushed to branch: {branch_to_use}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1) 