MSG := "default commit message"

build:
	hugo -t hugo-coder
	cd public && git add . && git commit -m "$(MSG)" && git push origin main 
