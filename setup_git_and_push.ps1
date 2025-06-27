
# Ask for GitHub repo URL
$repoURL = Read-Host "Enter your GitHub repository URL (e.g. https://github.com/yourusername/cereal-market-dashboard.git)"

# Confirm folder path
$projectPath = Read-Host "Enter the full path to your project folder (e.g. C:\Projects\cereal-dashboard)"

# Navigate to your project folder
cd $projectPath

# Initialize git
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit of cereal dashboard project"

# Add remote origin
git remote add origin $repoURL

# Set main branch and push
git branch -M main
git push -u origin main
